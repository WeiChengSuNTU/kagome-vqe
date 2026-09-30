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
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Final

import rustworkx as rx

__all__ = [
    "GUADALUPE_HEAVY_HEX_LOOP",
    "KAGOME_HAMILTONIAN_CYCLE",
    "KAGOME_UNIT_CELL_EDGES",
    "Edge",
    "Lattice",
    "inflated_guadalupe_16",
    "kagome_unit_cell",
    "perfect_matchings",
]

Edge = tuple[int, int, float]
"""A weighted undirected bond ``(site_a, site_b, weight)``."""

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
    """

    num_sites: int
    edges: tuple[Edge, ...]
    site_order: tuple[int, ...] | None = None
    name: str = "lattice"
    _adjacency: dict[int, frozenset[int]] = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        """Validate the bond list and cache the adjacency map."""
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
        object.__setattr__(self, "_adjacency", {k: frozenset(v) for k, v in adjacency.items()})

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
        return Lattice(
            num_sites=self.num_sites,
            edges=self.edges,
            site_order=None if site_order is None else tuple(site_order),
            name=self.name,
        )

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
