from collections import Counter

import pytest
import rustworkx as rx

from kagome_vqe.lattice import (
    KAGOME_HAMILTONIAN_CYCLE,
    Lattice,
    inflated_guadalupe_16,
    perfect_matchings,
)


def test_kagome_cell_geometry(lattice: Lattice) -> None:
    assert lattice.num_sites == 12
    assert lattice.num_bonds == 18
    degrees = Counter(lattice.degree(s) for s in range(12))
    assert degrees == {2: 6, 4: 6}
    assert all(w == 1.0 for _, _, w in lattice.edges)


def test_kagome_cell_is_six_corner_sharing_triangles(lattice: Lattice) -> None:
    graph = lattice.to_rustworkx()
    # every bond of a triangle sees exactly one common neighbour; 6 triangles x 3 bonds
    triangles = sum(len(lattice.neighbours(a) & lattice.neighbours(b)) for a, b, _ in lattice.edges)
    assert triangles == 18
    assert rx.is_connected(graph)


def test_hamiltonian_cycle_is_a_cycle(lattice: Lattice) -> None:
    order = KAGOME_HAMILTONIAN_CYCLE
    assert sorted(order) == list(range(12))
    for k in range(12):
        assert lattice.is_bond(order[k], order[(k + 1) % 12])
    assert lattice.ring_bond_fraction() == 1.0


def test_identity_order_is_a_poor_ring(identity_lattice: Lattice) -> None:
    assert identity_lattice.ring_bond_fraction() == pytest.approx(0.5)


def test_qubit_edges_apply_site_order(lattice: Lattice) -> None:
    assert lattice.site_order is not None
    inverse = lattice.qubit_of_site
    for a, b, _ in lattice.qubit_edges():
        assert lattice.is_bond(lattice.site_order[a], lattice.site_order[b])
    assert all(inverse[lattice.site_order[q]] == q for q in range(12))


def test_with_site_order_round_trip(lattice: Lattice) -> None:
    same = lattice.with_site_order(None).with_site_order(lattice.site_order)
    assert same == lattice


@pytest.mark.parametrize(
    ("edges", "message"),
    [
        (((0, 0, 1.0),), "self-loop"),
        (((0, 1, 1.0), (1, 0, 1.0)), "duplicate"),
        (((0, 5, 1.0),), "outside"),
        (((0, 1, 0.0),), "zero weight"),
    ],
)
def test_invalid_edges_rejected(edges: tuple[tuple[int, int, float], ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Lattice(num_sites=3, edges=edges)


def test_invalid_site_order_rejected() -> None:
    with pytest.raises(ValueError, match="permutation"):
        Lattice(num_sites=2, edges=((0, 1, 1.0),), site_order=(0, 0))


def test_two_dimer_coverings(lattice: Lattice) -> None:
    coverings = perfect_matchings(lattice)
    assert len(coverings) == 2
    spokes = {frozenset(b) for b in [(0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11)]}
    assert spokes in [{frozenset(b) for b in cover} for cover in coverings]


def test_legacy_16_qubit_lattice_matches_kagome_cell(lattice: Lattice) -> None:
    legacy = inflated_guadalupe_16()
    assert legacy.isolated_sites() == (0, 6, 9, 15)
    assert legacy.is_isomorphic_to(lattice)
    mapping = lattice.isomorphism_to(legacy)
    assert mapping is not None
    for a, b, _ in lattice.edges:
        assert legacy.is_bond(mapping[a], mapping[b])


def test_original_notebook_layout_was_not_faithful(lattice: Lattice) -> None:
    """The 2022 notebook's initial_layout maps only 2 of 18 bonds onto its 16-qubit lattice."""
    legacy = inflated_guadalupe_16()
    q_layout = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14]
    hits = sum(legacy.is_bond(q_layout[a], q_layout[b]) for a, b, _ in lattice.edges)
    assert hits == 2
