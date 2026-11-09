"""Spin lattices as plain, immutable graphs.

A :class:`Lattice` is nothing more than a set of sites and weighted bonds. It is deliberately
independent of Qiskit so that the geometry can be unit-tested, drawn and reasoned about without
any quantum machinery. :mod:`kagome_vqe.hamiltonian` turns a lattice into an operator.

Site ordering
-------------
The ansatzes in :mod:`kagome_vqe.ansatz` act on a *ring* of qubits: Bell pairs on ``(2k, 2k+1)``
and bond gates on ``(i, i+1)`` around the ring. They only work well when consecutive qubits host
neighbouring lattice sites. ``Lattice.site_order`` records which lattice site each ansatz qubit
hosts, so the mapping is explicit rather than an accident of transpilation (which is what
happened in the original 2022 notebook; see ``docs/physics.md``).

Geometry
--------
A lattice may optionally carry integer *triangular-lattice coordinates* ``coords[s] = (r, c)``
for every site. The kagome lattice is the triangular lattice (bonds ``(r, c)-(r, c+1)``,
``(r, c)-(r+1, c)`` and ``(r, c)-(r+1, c+1)``) with every site at odd ``r`` *and* odd ``c``
removed; those holes are the hexagon centres. The coordinates are only *metadata*: the bond
list alone defines the Hamiltonian and the logical ansatz, and the geometry is used by the
paper-inspired hardware mappings (:mod:`kagome_vqe.mapping`) and by plotting. Generators live in
:mod:`kagome_vqe.kagome_lattices`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Final, Literal

import rustworkx as rx

__all__ = [
    "BOUNDARIES",
    "GUADALUPE_HEAVY_HEX_LOOP",
    "KAGOME_HAMILTONIAN_CYCLE",
    "KAGOME_UNIT_CELL_COORDS",
    "KAGOME_UNIT_CELL_DIMERS",
    "KAGOME_UNIT_CELL_EDGES",
    "Boundary",
    "Edge",
    "Lattice",
    "count_dimer_coverings",
    "find_dimer_covering",
    "inflated_guadalupe_16",
    "is_kagome_hole",
    "is_kagome_neighbour",
    "kagome_unit_cell",
    "perfect_matchings",
    "validate_dimer_covering",
]

Edge = tuple[int, int, float]
"""A weighted undirected bond ``(site_a, site_b, weight)``."""

Boundary = Literal["open", "periodic", "unspecified"]
"""Boundary condition of a lattice. ``"unspecified"`` is for custom graphs without geometry."""

BOUNDARIES: Final[tuple[str, ...]] = ("open", "periodic", "unspecified")

KAGOME_UNIT_CELL_EDGES: Final[tuple[tuple[int, int], ...]] = (
    # inner hexagon
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
    # six outer "star" tips, each closing a triangle with one hexagon edge
    (0, 6), (1, 6),
    (1, 7), (2, 7),
    (2, 8), (3, 8),
    (3, 9), (4, 9),
    (4, 10), (5, 10),
    (5, 11), (0, 11),
)  # fmt: skip
"""The 18 bonds of the 12-site kagome unit cell used by the IBM Open Science Prize 2022."""

KAGOME_HAMILTONIAN_CYCLE: Final[tuple[int, ...]] = (0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11)
"""A Hamiltonian cycle of :data:`KAGOME_UNIT_CELL_EDGES`.

Reading the tuple cyclically, every consecutive pair is a lattice bond. Laying the ring ansatzes
along this order makes all 12 ring bonds *and* all 6 Bell pairs genuine lattice bonds, which is
the condition under which the RVB/HVA circuits reach the exact ground state. This is the default
``site_order`` of :func:`kagome_unit_cell`.
"""

KAGOME_UNIT_CELL_COORDS: Final[tuple[tuple[int, int], ...]] = (
    # hexagon 0-5 around the hole (3, 3)
    (2, 2), (2, 3), (3, 4), (4, 4), (4, 3), (3, 2),
    # tips 6-11, tip 6 caps hexagon edge (0, 1), tip 7 caps (1, 2), ...
    (1, 2), (2, 4), (4, 5), (5, 4), (4, 2), (2, 1),
)  # fmt: skip
"""Triangular-lattice coordinates of the 12 star sites (metadata for geometric mappings)."""

KAGOME_UNIT_CELL_DIMERS: Final[tuple[tuple[int, int], ...]] = (
    (0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11),
)  # fmt: skip
"""The spoke covering that the ring ansatzes prepare (one of the star's two exact ground states).

It equals the even ring bonds laid along :data:`KAGOME_HAMILTONIAN_CYCLE`.
"""

GUADALUPE_HEAVY_HEX_LOOP: Final[tuple[int, ...]] = (1, 2, 3, 5, 8, 11, 14, 13, 12, 10, 7, 4)
"""The 12-qubit loop of ``ibmq_guadalupe``'s heavy-hex coupling map.

Passing it as ``initial_layout`` places the ring ansatz on physically coupled qubits so that no
routing SWAPs are needed for the ring gates.
"""

_INFLATED_GUADALUPE_EDGES: Final[tuple[tuple[int, int], ...]] = (
    (1, 2), (2, 3), (3, 5), (5, 8), (8, 11), (11, 14),
    (14, 13), (13, 12), (12, 10), (10, 7), (7, 4), (4, 1),
    (4, 2), (2, 5), (5, 11), (11, 13), (13, 10), (10, 4),
)  # fmt: skip


@dataclass(frozen=True, slots=True)
class Lattice:
    """An undirected, weighted graph of spin-1/2 sites.

    Attributes:
        num_sites: Number of sites (one qubit each after mapping).
        edges: Weighted bonds. Each unordered pair appears once.
        site_order: ``site_order[k]`` is the lattice site hosted by ansatz qubit ``k``. ``None``
            means the identity (qubit ``k`` hosts site ``k``). It must be a permutation of
            ``range(num_sites)``.
        name: Human-readable label used in logs and result files.
        coords: Optional integer triangular-lattice coordinates ``(r, c)`` per site (see the
            module docstring). Pure metadata for geometric mappings and plots.
        boundary: ``"open"``, ``"periodic"`` or ``"unspecified"`` (custom graph).
        wrapped_bonds: Bonds (site pairs) that cross a periodic boundary. Must be bonds of
            the lattice, and empty unless ``boundary == "periodic"``.
        dimers: Optional preferred dimer covering (site pairs): a perfect matching made of
            lattice bonds, used as the singlet-product reference state of graph-based ansatzes.
    """

    num_sites: int
    edges: tuple[Edge, ...]
    site_order: tuple[int, ...] | None = None
    name: str = "lattice"
    coords: tuple[tuple[int, int], ...] | None = None
    boundary: Boundary = "open"
    wrapped_bonds: tuple[tuple[int, int], ...] = ()
    dimers: tuple[tuple[int, int], ...] | None = None
    _adjacency: dict[int, frozenset[int]] = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        """Validate the bond list, geometry and dimers, and cache the adjacency map."""
        if self.num_sites < 1:
            msg = f"a lattice needs at least one site, got num_sites={self.num_sites}"
            raise ValueError(msg)
        seen: set[frozenset[int]] = set()
        adjacency: dict[int, set[int]] = {i: set() for i in range(self.num_sites)}
        for a, b, w in self.edges:
            if not (0 <= a < self.num_sites and 0 <= b < self.num_sites):
                msg = f"edge ({a}, {b}) references a site outside range({self.num_sites})"
                raise ValueError(msg)
            if a == b:
                msg = f"self-loop on site {a} is not a bond"
                raise ValueError(msg)
            key = frozenset((a, b))
            if key in seen:
                msg = f"duplicate bond ({a}, {b})"
                raise ValueError(msg)
            seen.add(key)
            if w == 0:
                msg = f"bond ({a}, {b}) has zero weight"
                raise ValueError(msg)
            adjacency[a].add(b)
            adjacency[b].add(a)
        if self.site_order is not None and sorted(self.site_order) != list(range(self.num_sites)):
            msg = (
                f"site_order must be a permutation of range({self.num_sites}), "
                f"got {self.site_order}"
            )
            raise ValueError(msg)
        self._validate_metadata(seen)
        object.__setattr__(self, "_adjacency", {k: frozenset(v) for k, v in adjacency.items()})

    def _validate_metadata(self, bonds: set[frozenset[int]]) -> None:
        """Check boundary, coordinates, wrapped bonds and dimers against the bond set."""
        if self.boundary not in BOUNDARIES:
            msg = f"boundary must be one of {BOUNDARIES}, got {self.boundary!r}"
            raise ValueError(msg)
        if self.coords is not None:
            _validate_coords(self.coords, self.num_sites)
        for a, b in self.wrapped_bonds:
            if frozenset((a, b)) not in bonds:
                msg = f"wrapped bond ({a}, {b}) is not a bond of the lattice"
                raise ValueError(msg)
        if self.wrapped_bonds and self.boundary != "periodic":
            msg = f"wrapped_bonds given but boundary is {self.boundary!r}, not 'periodic'"
            raise ValueError(msg)
        if self.dimers is not None:
            validate_dimer_covering(self.dimers, self.num_sites, bonds)

    # -- geometry -----------------------------------------------------------------------------

    @property
    def num_bonds(self) -> int:
        """Number of bonds."""
        return len(self.edges)

    def neighbours(self, site: int) -> frozenset[int]:
        """Sites bonded to ``site``."""
        return self._adjacency[site]

    def degree(self, site: int) -> int:
        """Number of bonds touching ``site``."""
        return len(self._adjacency[site])

    def is_bond(self, a: int, b: int) -> bool:
        """Whether sites ``a`` and ``b`` share a bond."""
        return b in self._adjacency[a]

    # -- qubit labelling ----------------------------------------------------------------------

    @property
    def qubit_of_site(self) -> tuple[int, ...]:
        """Inverse of ``site_order``: ``qubit_of_site[s]`` is the qubit hosting site ``s``."""
        if self.site_order is None:
            return tuple(range(self.num_sites))
        inverse = [0] * self.num_sites
        for qubit, site in enumerate(self.site_order):
            inverse[site] = qubit
        return tuple(inverse)

    def qubit_edges(self) -> tuple[Edge, ...]:
        """Bonds expressed in ansatz-qubit labels (``site_order`` applied)."""
        q = self.qubit_of_site
        return tuple((q[a], q[b], w) for a, b, w in self.edges)

    def with_site_order(self, site_order: Sequence[int] | None) -> Lattice:
        """Return a copy hosting the same graph on a different qubit ordering."""
        return replace(self, site_order=None if site_order is None else tuple(site_order))

    def with_dimers(self, dimers: Sequence[tuple[int, int]] | None) -> Lattice:
        """Return a copy with a different preferred dimer covering (validated)."""
        return replace(
            self, dimers=None if dimers is None else tuple((int(a), int(b)) for a, b in dimers)
        )

    def qubit_pairs(self, pairs: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
        """Translate site pairs to ansatz-qubit pairs (``site_order`` applied)."""
        q = self.qubit_of_site
        return tuple((q[a], q[b]) for a, b in pairs)

    def bond_weight(self, a: int, b: int) -> float:
        """Coupling ``J_ab`` of the bond between sites ``a`` and ``b``.

        Raises:
            KeyError: If the sites share no bond.
        """
        for x, y, w in self.edges:
            if {x, y} == {a, b}:
                return w
        msg = f"({a}, {b}) is not a bond"
        raise KeyError(msg)

    def is_connected(self) -> bool:
        """Whether every site can be reached from every other along bonds."""
        if self.num_sites == 1:
            return True
        return bool(rx.is_connected(self.to_rustworkx()))

    @property
    def has_geometry(self) -> bool:
        """Whether triangular-lattice coordinates are attached."""
        return self.coords is not None

    def is_kagome_geometry(self) -> bool:
        """Whether ``coords`` exist and every bond is a nearest-neighbour kagome bond in them.

        This is what the geometric (paper-inspired) hardware mappings require. Wrapped
        (periodic) bonds are never nearest neighbours in the plane, so periodic lattices fail.
        """
        if self.coords is None:
            return False
        return all(is_kagome_neighbour(self.coords[a], self.coords[b]) for a, b, _ in self.edges)

    def ring_bond_fraction(self) -> float:
        """Fraction of the ansatz ring bonds ``(k, k+1 mod n)`` that are lattice bonds.

        A diagnostic for how well ``site_order`` suits the ring ansatzes: 1.0 means the ring
        is a Hamiltonian cycle of the lattice.
        """
        order = self.site_order or tuple(range(self.num_sites))
        n = self.num_sites
        hits = sum(self.is_bond(order[k], order[(k + 1) % n]) for k in range(n))
        return hits / n

    # -- interop ------------------------------------------------------------------------------

    def to_rustworkx(self) -> rx.PyGraph:
        """Export as a ``rustworkx.PyGraph`` (site labels as node indices, weights as payload)."""
        graph: rx.PyGraph = rx.PyGraph(multigraph=False)
        graph.add_nodes_from(list(range(self.num_sites)))
        graph.add_edges_from([(a, b, w) for a, b, w in self.edges])
        return graph

    def isolated_sites(self) -> tuple[int, ...]:
        """Sites with no bonds (idle qubits, e.g. 0, 6, 9, 15 in the legacy 16-qubit lattice)."""
        return tuple(s for s in range(self.num_sites) if self.degree(s) == 0)

    def is_isomorphic_to(self, other: Lattice) -> bool:
        """Graph isomorphism ignoring weights, site order and isolated sites."""
        g1, g2 = self.to_rustworkx(), other.to_rustworkx()
        g1.remove_nodes_from(list(self.isolated_sites()))
        g2.remove_nodes_from(list(other.isolated_sites()))
        return bool(rx.is_isomorphic(g1, g2))

    def isomorphism_to(self, other: Lattice) -> dict[int, int] | None:
        """A site map ``self -> other`` realising an isomorphism, or ``None`` if there is none.

        Isolated sites are ignored on both sides.
        """
        g1, g2 = self.to_rustworkx(), other.to_rustworkx()
        g1.remove_nodes_from(list(self.isolated_sites()))
        g2.remove_nodes_from(list(other.isolated_sites()))
        mapping = rx.graph_vf2_mapping(g1, g2, subgraph=False, induced=True)
        found = next(iter(mapping), None)
        return None if found is None else dict(found)


def kagome_unit_cell(
    t: float = 1.0,
    *,
    site_order: Sequence[int] | None = KAGOME_HAMILTONIAN_CYCLE,
) -> Lattice:
    """The 12-site kagome "star" of the IBM Open Science Prize 2022.

    Six sites form a hexagon; each hexagon edge is capped by an outer tip, giving six corner-
    sharing triangles, 18 bonds, and degrees ``{2: 6, 4: 6}``. With open boundaries the ground
    state is an exact product of six singlets on the spokes (energy ``-18`` for ``t = 1``), and
    it is two-fold degenerate.

    Args:
        t: Uniform exchange coupling on every bond.
        site_order: Which lattice site each ansatz qubit hosts. The default lays the ring
            ansatzes along a Hamiltonian cycle (the physically sensible choice). Pass ``None``
            for the identity order, which is a deliberately hard problem for ring ansatzes.

    Returns:
        The lattice.
    """
    edges = tuple((a, b, float(t)) for a, b in KAGOME_UNIT_CELL_EDGES)
    return Lattice(
        num_sites=12,
        edges=edges,
        site_order=None if site_order is None else tuple(site_order),
        name="kagome_unit_cell",
        coords=KAGOME_UNIT_CELL_COORDS,
        boundary="open",
        dimers=KAGOME_UNIT_CELL_DIMERS,
    )


def inflated_guadalupe_16(t: float = 1.0) -> Lattice:
    """The legacy 16-qubit Hamiltonian of the original notebook, in physical-qubit labels.

    The 2022 challenge hand-wrote the kagome bonds directly on ``ibmq_guadalupe`` qubits, leaving
    qubits 0, 6, 9 and 15 idle. Kept as a documented fixture: it is graph-isomorphic to
    :func:`kagome_unit_cell` but is *not* the image of that lattice under the notebook's
    ``initial_layout``. New code should use :func:`kagome_unit_cell` and let
    :func:`kagome_vqe.transpile.compile_for_backend` apply the layout.
    """
    edges = tuple((a, b, float(t)) for a, b in _INFLATED_GUADALUPE_EDGES)
    return Lattice(num_sites=16, edges=edges, name="inflated_guadalupe_16")


def perfect_matchings(lattice: Lattice) -> list[tuple[tuple[int, int], ...]]:
    """All dimer coverings: bond sets touching every site exactly once.

    Brute force (fine for 12 sites, exponential in general). Used to explain the exact
    ground-state energy of the open kagome star and in tests.
    """
    bonds = [(a, b) for a, b, _ in lattice.edges]
    n_dimers = lattice.num_sites // 2
    if lattice.num_sites % 2:
        return []
    result: list[tuple[tuple[int, int], ...]] = []
    for combo in combinations(bonds, n_dimers):
        covered = [s for bond in combo for s in bond]
        if len(set(covered)) == lattice.num_sites:
            result.append(combo)
    return result


# -- geometry helpers -------------------------------------------------------------------------


def is_kagome_hole(coord: tuple[int, int]) -> bool:
    """Whether a triangular-lattice point is a hexagon centre: odd ``r`` and odd ``c``."""
    return coord[0] % 2 == 1 and coord[1] % 2 == 1


def is_kagome_neighbour(p: tuple[int, int], q: tuple[int, int]) -> bool:
    """Whether two triangular-lattice points are nearest neighbours (and neither is a hole).

    The triangular neighbours of ``(r, c)`` are ``(r, c+-1)``, ``(r+-1, c)`` and
    ``(r+-1, c+-1)`` (same sign).
    """
    if is_kagome_hole(p) or is_kagome_hole(q):
        return False
    dr, dc = q[0] - p[0], q[1] - p[1]
    return (dr, dc) in {(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1)}


def _validate_coords(coords: tuple[tuple[int, int], ...], num_sites: int) -> None:
    if len(coords) != num_sites:
        msg = f"coords has {len(coords)} entries for {num_sites} sites"
        raise ValueError(msg)
    seen: set[tuple[int, int]] = set()
    for site, coord in enumerate(coords):
        if len(coord) != 2 or not all(isinstance(x, int) for x in coord):
            msg = f"coords[{site}] = {coord!r} is not an integer pair (r, c)"
            raise ValueError(msg)
        if coord in seen:
            msg = f"two sites share the coordinate {coord}"
            raise ValueError(msg)
        if is_kagome_hole(coord):
            msg = f"coords[{site}] = {coord} is a hexagon centre (odd, odd), not a kagome site"
            raise ValueError(msg)
        seen.add(coord)


def validate_dimer_covering(
    dimers: Sequence[tuple[int, int]],
    num_sites: int,
    bonds: set[frozenset[int]] | None = None,
) -> None:
    """Check that ``dimers`` is a perfect matching (made of ``bonds`` if given).

    Raises:
        ValueError: With the first violation found: odd site count, a pair that is not a
            bond, a site covered twice, or an uncovered site.
    """
    if num_sites % 2:
        msg = f"a dimer covering needs an even number of sites, got {num_sites}"
        raise ValueError(msg)
    covered: set[int] = set()
    for a, b in dimers:
        if not (0 <= a < num_sites and 0 <= b < num_sites) or a == b:
            msg = f"dimer ({a}, {b}) is not a pair of distinct sites in range({num_sites})"
            raise ValueError(msg)
        if bonds is not None and frozenset((a, b)) not in bonds:
            msg = f"dimer ({a}, {b}) is not a bond of the lattice"
            raise ValueError(msg)
        for s in (a, b):
            if s in covered:
                msg = f"site {s} is covered by more than one dimer"
                raise ValueError(msg)
            covered.add(s)
    if len(covered) != num_sites:
        missing = sorted(set(range(num_sites)) - covered)
        msg = f"dimer covering leaves sites {missing} uncovered"
        raise ValueError(msg)


def find_dimer_covering(lattice: Lattice) -> tuple[tuple[int, int], ...]:
    """The lexicographically first perfect matching of the bond graph.

    Deterministic depth-first search: always cover the lowest-numbered free site, trying its
    neighbours in ascending order. Exponential in the worst case but immediate for the kagome
    patches used here (tens of sites).

    Raises:
        ValueError: If the graph has no perfect matching (e.g. an odd number of sites).
    """
    n = lattice.num_sites
    if n % 2:
        msg = f"{lattice.name}: {n} sites cannot be covered by dimers (odd site count)"
        raise ValueError(msg)
    free = [True] * n
    chosen: list[tuple[int, int]] = []

    def search() -> bool:
        try:
            site = free.index(True)
        except ValueError:
            return True
        free[site] = False
        for other in sorted(lattice.neighbours(site)):
            if free[other]:
                free[other] = False
                chosen.append((site, other))
                if search():
                    return True
                chosen.pop()
                free[other] = True
        free[site] = True
        return False

    if not search():
        msg = f"{lattice.name}: the bond graph has no perfect matching (no dimer covering)"
        raise ValueError(msg)
    return tuple(chosen)


def count_dimer_coverings(lattice: Lattice) -> int:
    """Number of perfect matchings (dimer coverings), by backtracking.

    Exact and fast for the kagome patches used here; exponential in general.
    """
    n = lattice.num_sites
    if n % 2:
        return 0
    free = [True] * n

    def count() -> int:
        try:
            site = free.index(True)
        except ValueError:
            return 1
        free[site] = False
        total = 0
        for other in lattice.neighbours(site):
            if free[other]:
                free[other] = False
                total += count()
                free[other] = True
        free[site] = True
        return total

    return count()
