"""Deterministic gate schedules for graph-based Hamiltonian variational ansatzes.

An :class:`ExchangeSchedule` lists the bonds of a lattice as an ordered sequence of *layers*.
Each layer is a matching (no two bonds share a site), so its exchange gates commute and can run
in parallel; every bond appears in exactly one layer. One HVA cycle applies the layers in order.

The schedule is purely logical. It fixes the *order* of the non-commuting exchange gates and
hence the variational state; it says nothing about qubit placement or SWAPs, which are the
business of :mod:`kagome_vqe.mapping`. A hardware mapping must reproduce this order exactly.

Schedules (all in *site* labels):

``greedy``
    A minimum (or near-minimum) edge colouring found by deterministic search, with colour
    classes ordered so that classes containing reference dimers come last (a gate on an
    untouched dimer only adds a global phase).
``prb105``
    The five groups ``H1 .. H5`` of Bosse & Montanaro, PRB 105, 094409, Fig. 2, in the order
    of their Eq. (4), ``U = prod_i [e^{-i theta_0 H0} U_3 U_2 U_1 U_5 U_4]``, read as an operator
    product (rightmost acts first): chronologically ``H4, H5, H1, H2, H3`` (then ``H0`` with
    ``h0_position = "end"``). With this reading no first-cycle gate acts on an untouched
    singlet and the H0 block never fuses with ``H1 = H0``. Needs kagome coordinates.
``prb105_fig3``
    The same groups in the listing order of their Fig. 3, ``H1, H2, H3, H4, H5``. Same circuit
    depth, but the first-cycle ``H1`` gates act on the reference singlets (global phase only).
``prb106``
    The six layers of the KVQE_G cycle of Kattemolle & van Wezel, PRB 106, 214429, Fig. 2.
    Needs kagome coordinates.
``explicit``
    User-supplied layers, validated.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, get_args

from kagome_vqe.kagome_lattices import bm_coord_to_grid
from kagome_vqe.lattice import Lattice

__all__ = [
    "Bond",
    "ExchangeSchedule",
    "ScheduleKind",
    "explicit_schedule",
    "greedy_schedule",
    "make_schedule",
    "prb105_schedule",
    "prb106_schedule",
]

Bond = tuple[int, int]
ScheduleKind = Literal["greedy", "prb105", "prb105_fig3", "prb106", "explicit"]

_MAX_COLOURING_STEPS = 200_000
"""Search budget for an edge colouring with the minimum number of colours."""


def _key(a: int, b: int) -> frozenset[int]:
    return frozenset((a, b))


@dataclass(frozen=True, slots=True)
class ExchangeSchedule:
    """Ordered layers of disjoint bonds covering a lattice once.

    Attributes:
        layers: ``layers[k]`` is a tuple of bonds ``(a, b)`` (site labels, ``a < b``) acting in
            parallel in step ``k`` of every cycle.
        kind: How the schedule was produced.
        labels: Human-readable name per layer (e.g. ``"H3"``).
    """

    layers: tuple[tuple[Bond, ...], ...]
    kind: str
    labels: tuple[str, ...] = ()

    @property
    def num_layers(self) -> int:
        """Number of layers (the two-qubit depth of one cycle on all-to-all hardware)."""
        return len(self.layers)

    @property
    def bonds(self) -> tuple[Bond, ...]:
        """All bonds in chronological order within one cycle."""
        return tuple(b for layer in self.layers for b in layer)

    def validate(self, lattice: Lattice) -> None:
        """Check that every layer is a matching and every bond occurs exactly once.

        Raises:
            ValueError: Naming the first offending layer or bond.
        """
        bonds = {_key(a, b) for a, b, _ in lattice.edges}
        seen: set[frozenset[int]] = set()
        for k, layer in enumerate(self.layers):
            sites: set[int] = set()
            for a, b in layer:
                key = _key(a, b)
                if key not in bonds:
                    msg = f"schedule layer {k}: ({a}, {b}) is not a bond of {lattice.name}"
                    raise ValueError(msg)
                if key in seen:
                    msg = f"schedule layer {k}: bond ({a}, {b}) is scheduled more than once"
                    raise ValueError(msg)
                if a in sites or b in sites:
                    msg = f"schedule layer {k}: bonds share a site, the layer is not a matching"
                    raise ValueError(msg)
                sites.update((a, b))
                seen.add(key)
        missing = bonds - seen
        if missing:
            example = sorted(tuple(sorted(m)) for m in missing)[:5]
            msg = f"schedule misses {len(missing)} bonds of {lattice.name}, e.g. {example}"
            raise ValueError(msg)

    def reordered(self, order: Sequence[int]) -> ExchangeSchedule:
        """The same layers applied in a different order (``order`` is a permutation).

        Example: ``prb105_schedule(lat).reordered([3, 4, 0, 1, 2])`` applies
        ``H4, H5, H1, H2, H3``, the chronological reading of PRB 105 Eq. (4).
        """
        if sorted(order) != list(range(self.num_layers)):
            msg = (
                f"layer order must be a permutation of range({self.num_layers}), got {list(order)}"
            )
            raise ValueError(msg)
        labels = tuple(self.labels[k] for k in order) if self.labels else ()
        return ExchangeSchedule(
            layers=tuple(self.layers[k] for k in order), kind=self.kind, labels=labels
        )

    def to_list(self) -> list[list[list[int]]]:
        """Plain nested lists (JSON/TOML friendly)."""
        return [[[a, b] for a, b in layer] for layer in self.layers]


def _normalised(layers: Sequence[Sequence[Bond]]) -> tuple[tuple[Bond, ...], ...]:
    return tuple(
        tuple(sorted((min(a, b), max(a, b)) for a, b in layer)) for layer in layers if layer
    )


def _colour_edges(lattice: Lattice, num_colours: int, budget: int) -> list[int] | None:
    """Backtracking proper edge colouring with ``num_colours`` colours, or ``None``."""
    edges = [(min(a, b), max(a, b)) for a, b, _ in lattice.edges]
    order = sorted(range(len(edges)), key=lambda i: edges[i])
    colour = [-1] * len(edges)
    used: dict[int, set[int]] = {s: set() for s in range(lattice.num_sites)}
    steps = 0

    def place(pos: int) -> bool:
        nonlocal steps
        if pos == len(order):
            return True
        steps += 1
        if steps > budget:
            return False
        i = order[pos]
        a, b = edges[i]
        for col in range(num_colours):
            if col in used[a] or col in used[b]:
                continue
            colour[i] = col
            used[a].add(col)
            used[b].add(col)
            if place(pos + 1):
                return True
            used[a].discard(col)
            used[b].discard(col)
            colour[i] = -1
        return False

    return colour if place(0) else None


def greedy_schedule(
    lattice: Lattice,
    dimers: Sequence[Bond] | None = None,
    *,
    max_colours: int | None = None,
    include_h0: bool = False,
) -> ExchangeSchedule:
    """Deterministic edge colouring with the fewest colours found, as a schedule.

    Tries ``Delta`` colours (``Delta`` = maximum degree, a lower bound), then ``Delta + 1``
    (always possible by Vizing's theorem), each with a bounded backtracking search over bonds in
    sorted order. Colour classes are then ordered by how many reference dimers they contain
    (fewest first, ties by colour index), so that gates on still-untouched singlets, which only
    contribute a global phase, are pushed late in the first cycle. With ``include_h0`` the
    dimer-rich classes go to the *middle* instead (the two dimer-poorest at the ends), so the
    H0 block, which acts on the dimers, is not adjacent to them and does not fuse with them.

    Args:
        lattice: The interaction graph.
        dimers: Reference dimer covering used for the ordering heuristic (site labels).
        max_colours: Optional cap on the number of colours tried.
        include_h0: Order for an ansatz with an H0 block (see above).

    Raises:
        ValueError: If no colouring within the budget was found (not expected for kagome).
    """
    if lattice.num_bonds == 0:
        msg = f"{lattice.name} has no bonds to schedule"
        raise ValueError(msg)
    delta = max(lattice.degree(s) for s in range(lattice.num_sites))
    limit = max_colours or delta + 2
    edges = [(min(a, b), max(a, b)) for a, b, _ in lattice.edges]
    colouring: list[int] | None = None
    colours = delta
    for colours in range(delta, limit + 1):
        colouring = _colour_edges(lattice, colours, _MAX_COLOURING_STEPS)
        if colouring is not None:
            break
    if colouring is None:
        msg = f"no edge colouring of {lattice.name} with <= {limit} colours found"
        raise ValueError(msg)
    classes: list[list[Bond]] = [[] for _ in range(colours)]
    for edge, col in zip(edges, colouring, strict=True):
        classes[col].append(edge)
    dimer_keys = {_key(a, b) for a, b in dimers or ()}
    ranked = sorted(
        range(colours),
        key=lambda k: (sum(_key(*e) in dimer_keys for e in classes[k]), k),
    )
    if include_h0 and len(ranked) > 2:
        # ascending a0 <= a1 <= ... -> a0, a2, a4, ..., a5, a3, a1: smallest at both ends
        ranked = ranked[0::2] + ranked[1::2][::-1]
    layers = _normalised([classes[k] for k in ranked])
    schedule = ExchangeSchedule(
        layers=layers, kind="greedy", labels=tuple(f"colour{k}" for k in range(len(layers)))
    )
    schedule.validate(lattice)
    return schedule


def _require_planar_kagome(lattice: Lattice, what: str) -> tuple[tuple[int, int], ...]:
    if lattice.boundary == "periodic" or not lattice.is_kagome_geometry():
        msg = (
            f"the {what} schedule needs an open kagome patch with triangular-lattice coordinates "
            f"in which every bond is a nearest neighbour; {lattice.name} "
            f"(boundary={lattice.boundary!r}, coords={'yes' if lattice.coords else 'no'}) "
            "does not qualify"
        )
        raise ValueError(msg)
    assert lattice.coords is not None
    return lattice.coords


def prb106_schedule(lattice: Lattice) -> ExchangeSchedule:
    """The six-layer KVQE_G cycle of PRB 106, Fig. 2, by bond orientation and parity.

    With bonds written ``(r, c) -> (r + dr, c + dc)``, ``dr, dc >= 0``:

    1. horizontal ``(0, 1)`` with odd ``c``,   2. diagonal ``(1, 1)`` with even ``r``,
    3. horizontal with even ``c``,             4. vertical ``(1, 0)`` with even ``r``,
    5. diagonal with odd ``r``,                6. vertical with odd ``r``.

    On the paper's 20-site patch these are exactly the coloured gates of panels (1)-(6). The
    same parity rules define a valid schedule on any open patch.
    """
    coords = _require_planar_kagome(lattice, "prb106")
    layers: list[list[Bond]] = [[] for _ in range(6)]
    for a, b, _ in lattice.edges:
        p, q = (coords[a], coords[b]) if coords[a] <= coords[b] else (coords[b], coords[a])
        (r, c), (dr, dc) = p, (q[0] - p[0], q[1] - p[1])
        if (dr, dc) == (0, 1):
            k = 0 if c % 2 else 2
        elif (dr, dc) == (1, 1):
            k = 1 if r % 2 == 0 else 4
        elif (dr, dc) == (1, 0):
            k = 3 if r % 2 == 0 else 5
        else:  # pragma: no cover - excluded by _require_planar_kagome
            msg = f"bond ({a}, {b}) is not a kagome nearest neighbour"
            raise ValueError(msg)
        layers[k].append((min(a, b), max(a, b)))
    labels = ("L1", "L2", "L3", "L4", "L5", "L6")
    kept = [(lab, lay) for lab, lay in zip(labels, layers, strict=True) if lay]
    schedule = ExchangeSchedule(
        layers=_normalised([lay for _, lay in kept]),
        kind="prb106",
        labels=tuple(lab for lab, _ in kept),
    )
    schedule.validate(lattice)
    return schedule


def prb105_schedule(lattice: Lattice, *, order: Literal["eq4", "fig3"] = "eq4") -> ExchangeSchedule:
    """The five interaction groups of PRB 105, Fig. 2.

    Bonds are classified by their displacement in the square-grid embedding
    (:func:`~kagome_vqe.kagome_lattices.bm_coord_to_grid`):

    * ``H1`` / ``H2``: horizontal grid neighbours, left column even / odd;
    * ``H3``: vertical grid neighbours;
    * ``H4``: the remaining up-right diagonal bond of each apex site (needs a SWAP);
    * ``H5``: next-nearest horizontal bonds across an apex site (needs a SWAP).

    ``order="eq4"`` (default) applies them as ``H4, H5, H1, H2, H3`` (Eq. (4) read as an
    operator product); ``order="fig3"`` as ``H1, H2, H3, H4, H5`` (the listing of Fig. 3).
    Both need one SWAP layer before ``H4``/``H5`` and one after, i.e. 7 layers per cycle.
    """
    coords = _require_planar_kagome(lattice, "prb105")
    grid = [bm_coord_to_grid(c) for c in coords]
    layers: list[list[Bond]] = [[] for _ in range(5)]
    for a, b, _ in lattice.edges:
        (ga, ja), (gb, jb) = grid[a], grid[b]
        if (ga, ja) > (gb, jb):
            (ga, ja), (gb, jb) = (gb, jb), (ga, ja)
        d = (gb - ga, jb - ja)
        if d == (0, 1):
            k = 0 if ja % 2 == 0 else 1
        elif d == (1, 0):
            k = 2
        elif d == (1, -1):
            k = 3
        elif d == (0, 2):
            k = 4
        else:  # pragma: no cover - geometry guarantees one of the above
            msg = f"bond ({a}, {b}) has unexpected grid displacement {d} in the PRB 105 embedding"
            raise ValueError(msg)
        layers[k].append((min(a, b), max(a, b)))
    labels = ("H1", "H2", "H3", "H4", "H5")
    sequence = (3, 4, 0, 1, 2) if order == "eq4" else (0, 1, 2, 3, 4)
    kept = [(labels[k], layers[k]) for k in sequence if layers[k]]
    schedule = ExchangeSchedule(
        layers=_normalised([lay for _, lay in kept]),
        kind="prb105" if order == "eq4" else "prb105_fig3",
        labels=tuple(lab for lab, _ in kept),
    )
    schedule.validate(lattice)
    return schedule


def explicit_schedule(
    lattice: Lattice, layers: Sequence[Sequence[Sequence[int]]]
) -> ExchangeSchedule:
    """Validate user-given layers (site labels) as a schedule."""
    parsed: list[list[Bond]] = []
    for k, layer in enumerate(layers):
        bonds: list[Bond] = []
        for pair in layer:
            if len(pair) != 2:
                msg = f"explicit schedule layer {k}: {list(pair)} is not a pair of sites"
                raise ValueError(msg)
            bonds.append((int(pair[0]), int(pair[1])))
        parsed.append(bonds)
    schedule = ExchangeSchedule(
        layers=_normalised(parsed),
        kind="explicit",
        labels=tuple(f"layer{k}" for k in range(len(parsed))),
    )
    schedule.validate(lattice)
    return schedule


def make_schedule(
    lattice: Lattice,
    kind: ScheduleKind,
    *,
    dimers: Sequence[Bond] | None = None,
    layers: Sequence[Sequence[Sequence[int]]] | None = None,
    include_h0: bool = False,
) -> ExchangeSchedule:
    """Dispatch on ``kind``; ``layers`` is required for (and only used by) ``"explicit"``.

    ``dimers`` and ``include_h0`` only steer the layer order of ``"greedy"``.
    """
    if kind not in get_args(ScheduleKind):
        msg = f"unknown schedule kind {kind!r}; choose from {get_args(ScheduleKind)}"
        raise ValueError(msg)
    if kind == "greedy":
        return greedy_schedule(lattice, dimers, include_h0=include_h0)
    if kind == "prb105":
        return prb105_schedule(lattice)
    if kind == "prb105_fig3":
        return prb105_schedule(lattice, order="fig3")
    if kind == "prb106":
        return prb106_schedule(lattice)
    if layers is None:  # kind == "explicit"
        msg = "schedule kind 'explicit' needs the layers"
        raise ValueError(msg)
    return explicit_schedule(lattice, layers)
