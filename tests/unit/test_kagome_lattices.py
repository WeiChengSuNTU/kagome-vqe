"""Generalised lattices: generators, validation, boundary conditions and Hamiltonians."""

from __future__ import annotations

import numpy as np
import pytest

from kagome_vqe.hamiltonian import exact_spectrum, heisenberg_hamiltonian
from kagome_vqe.kagome_lattices import (
    KVQE20_COORDS,
    bm_coord_to_grid,
    bm_grid_to_coord,
    custom_lattice,
    kagome_bm_rectangle,
    kagome_from_coords,
    kagome_kvqe20,
    kagome_patch,
    kagome_periodic,
)
from kagome_vqe.lattice import (
    KAGOME_UNIT_CELL_EDGES,
    Lattice,
    find_dimer_covering,
    is_kagome_neighbour,
    kagome_unit_cell,
    validate_dimer_covering,
)


def _bond_set(lattice: Lattice) -> set[frozenset[int]]:
    return {frozenset((a, b)) for a, b, _ in lattice.edges}


# -- the original 12-site cell is preserved -------------------------------------------------


def test_unit_cell_bonds_unchanged() -> None:
    lat = kagome_unit_cell()
    assert _bond_set(lat) == {frozenset(e) for e in KAGOME_UNIT_CELL_EDGES}
    assert lat.num_bonds == 18
    assert lat.boundary == "open"


def test_unit_cell_coordinates_reproduce_its_bonds() -> None:
    lat = kagome_unit_cell()
    assert lat.is_kagome_geometry()
    regenerated = kagome_from_coords(lat.coords or ())
    assert _bond_set(regenerated) == _bond_set(lat)


def test_unit_cell_hamiltonian_and_spectrum_unchanged() -> None:
    lat = kagome_unit_cell()
    hamiltonian = heisenberg_hamiltonian(lat)
    assert len(hamiltonian) == 54
    np.testing.assert_allclose(
        exact_spectrum(hamiltonian, k=3), [-18, -18, -16.96132284], atol=1e-7
    )


def test_unit_cell_default_dimers_are_the_ring_singlets() -> None:
    lat = kagome_unit_cell()
    order = lat.site_order
    assert order is not None
    ring_pairs = {frozenset((order[2 * k], order[2 * k + 1])) for k in range(6)}
    assert {frozenset(d) for d in lat.dimers or ()} == ring_pairs


# -- generators -----------------------------------------------------------------------------


def test_kvqe20_matches_paper_counts() -> None:
    lat = kagome_kvqe20()
    assert (lat.num_sites, lat.num_bonds) == (20, 30)  # PRB 106: 20 sites, 30 HEIS gates/cycle
    assert lat.dimers is not None
    assert len(lat.dimers) == 10
    assert lat.coords == KVQE20_COORDS


@pytest.mark.parametrize(("rows", "cols"), [(2, 4), (2, 6), (2, 8), (3, 6), (2, 10), (3, 8)])
def test_bm_rectangles_are_kagome(rows: int, cols: int) -> None:
    lat = kagome_bm_rectangle(rows, cols)
    assert lat.num_sites == rows * cols
    assert lat.is_kagome_geometry()
    assert lat.is_connected()
    assert max(lat.degree(s) for s in range(lat.num_sites)) <= 4
    assert lat.dimers is not None  # even cols: the colour-1 pairs form a perfect matching


def test_bm_odd_columns_have_no_default_covering() -> None:
    assert kagome_bm_rectangle(2, 5).dimers is None


def test_bm_coordinate_maps_are_inverse() -> None:
    for g in range(-4, 5):
        for j in range(-7, 8):
            assert bm_coord_to_grid(bm_grid_to_coord(g, j)) == (g, j)


def test_bm_bonds_are_short_on_the_grid() -> None:
    """Every kagome bond is a grid edge, a grid diagonal or a next-nearest horizontal pair."""
    lat = kagome_bm_rectangle(3, 8)
    assert lat.coords is not None
    for a, b, _ in lat.edges:
        (ga, ja), (gb, jb) = bm_coord_to_grid(lat.coords[a]), bm_coord_to_grid(lat.coords[b])
        assert (abs(gb - ga), abs(jb - ja)) in {(0, 1), (1, 0), (1, 1), (0, 2)}


@pytest.mark.parametrize(("l1", "l2"), [(2, 2), (2, 3), (3, 3)])
def test_periodic_torus_is_four_regular(l1: int, l2: int) -> None:
    lat = kagome_periodic(l1, l2)
    assert lat.num_sites == 3 * l1 * l2
    assert lat.num_bonds == 6 * l1 * l2
    assert all(lat.degree(s) == 4 for s in range(lat.num_sites))
    assert lat.boundary == "periodic"
    assert lat.wrapped_bonds
    assert not lat.is_kagome_geometry()  # wrapped bonds are not planar
    # every site sits in exactly two triangles on the torus
    triangles = sum(1 for a, b, _ in lat.edges for c in lat.neighbours(a) & lat.neighbours(b)) // 3
    assert triangles == 2 * lat.num_sites // 3


def test_periodic_torus_ground_energy() -> None:
    """12-site kagome torus: E0/N = -0.45374 J in S.S units (4x in Pauli units)."""
    energy = exact_spectrum(heisenberg_hamiltonian(kagome_periodic(2, 2)), k=1)[0]
    assert energy / 4 / 12 == pytest.approx(-0.453742, abs=1e-5)


def test_periodic_rejects_too_small_torus() -> None:
    with pytest.raises(ValueError, match="l1, l2 >= 2"):
        kagome_periodic(1, 3)


def test_patch_window_and_drop() -> None:
    lat = kagome_patch(5, 5, drop=[(0, 4)])
    assert {tuple(c) for c in lat.coords or ()} == set(KVQE20_COORDS)


def test_patch_rejects_disconnected() -> None:
    with pytest.raises(ValueError, match="not connected"):
        kagome_from_coords([(0, 0), (0, 4)])


# -- custom graphs and validation -----------------------------------------------------------


def test_custom_lattice_with_weights() -> None:
    lat = custom_lattice(4, [[0, 1], [1, 2, 0.5], [2, 3], [3, 0, 2.0]], coupling=1.5)
    assert lat.bond_weight(0, 1) == 1.5
    assert lat.bond_weight(2, 1) == 0.5
    hamiltonian = heisenberg_hamiltonian(lat)
    coeffs = dict(zip(hamiltonian.paulis.to_labels(), hamiltonian.coeffs, strict=True))
    assert coeffs["IIZZ"] == pytest.approx(1.5)  # bond (0, 1), default coupling
    assert coeffs["IZZI"] == pytest.approx(0.5)  # bond (1, 2)
    assert coeffs["ZIIZ"] == pytest.approx(2.0)  # bond (3, 0)


@pytest.mark.parametrize(
    ("bonds", "message"),
    [
        ([[0, 1], [1, 0]], "duplicate bond"),
        ([[0, 0]], "self-loop"),
        ([[0, 5]], "outside range"),
        ([[0, 1, 0.0]], "zero weight"),
        ([[0, 1, 2, 3]], r"\[a, b\] or \[a, b, J\]"),
    ],
)
def test_custom_lattice_rejects_bad_bonds(bonds: list[list[float]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        custom_lattice(4, bonds, require_connected=False)


def test_custom_lattice_rejects_disconnected_graph() -> None:
    with pytest.raises(ValueError, match="not connected"):
        custom_lattice(4, [[0, 1], [2, 3]])


def test_custom_coords_must_match_bonds() -> None:
    with pytest.raises(ValueError, match="not kagome nearest neighbours"):
        custom_lattice(2, [[0, 1]], coords=[[0, 0], [0, 2]], boundary="open")


def test_coords_reject_hexagon_centres() -> None:
    with pytest.raises(ValueError, match="hexagon centre"):
        Lattice(num_sites=2, edges=((0, 1, 1.0),), coords=((0, 0), (1, 1)))


def test_wrapped_bonds_need_periodic_boundary() -> None:
    with pytest.raises(ValueError, match="not 'periodic'"):
        Lattice(num_sites=2, edges=((0, 1, 1.0),), wrapped_bonds=((0, 1),))


def test_dimer_validation() -> None:
    lat = kagome_unit_cell()
    with pytest.raises(ValueError, match="not a bond"):
        lat.with_dimers([(0, 3), (1, 2), (4, 5), (6, 7), (8, 9), (10, 11)])
    with pytest.raises(ValueError, match="uncovered"):
        lat.with_dimers([(0, 6)])
    with pytest.raises(ValueError, match="more than one dimer"):
        validate_dimer_covering([(0, 1), (1, 2)], 4)
    with pytest.raises(ValueError, match="odd"):
        find_dimer_covering(custom_lattice(3, [[0, 1], [1, 2]]))


def test_find_dimer_covering_is_deterministic_and_valid() -> None:
    lat = kagome_periodic(2, 3)
    first = find_dimer_covering(lat)
    assert first == find_dimer_covering(lat)
    validate_dimer_covering(first, lat.num_sites, _bond_set(lat))


def test_graph_without_perfect_matching_is_rejected() -> None:
    star = custom_lattice(4, [[0, 1], [0, 2], [0, 3]])
    with pytest.raises(ValueError, match="no perfect matching"):
        find_dimer_covering(star)


def test_site_order_and_geometry_survive_copies() -> None:
    lat = kagome_bm_rectangle(2, 4)
    moved = lat.with_site_order(list(reversed(range(8))))
    assert moved.coords == lat.coords
    assert moved.dimers == lat.dimers
    assert sorted(exact_spectrum(heisenberg_hamiltonian(moved), k=1)) == pytest.approx(
        sorted(exact_spectrum(heisenberg_hamiltonian(lat), k=1))
    )


def test_neighbour_predicate() -> None:
    assert is_kagome_neighbour((0, 0), (1, 1)) is False  # (1, 1) is a hole
    assert is_kagome_neighbour((0, 1), (1, 2))
    assert not is_kagome_neighbour((0, 0), (1, -1))  # anti-diagonal is not a bond


def test_degenerate_ground_space_is_orthonormal() -> None:
    """The star's two-fold ground space; fidelity of a dimer product with it is exactly 1."""
    from qiskit.quantum_info import Statevector

    from kagome_vqe.ansatz import GraphHVAnsatz
    from kagome_vqe.hamiltonian import exact_ground_space, ground_state_fidelity

    lat = kagome_unit_cell()
    energy, space = exact_ground_space(heisenberg_hamiltonian(lat))
    assert energy == pytest.approx(-18)
    assert space.shape[1] == 2
    np.testing.assert_allclose(space.conj().T @ space, np.eye(2), atol=1e-12)
    circuit = GraphHVAnsatz(lat, 1).build()
    state = Statevector(circuit.assign_parameters(np.zeros(circuit.num_parameters))).data
    assert ground_state_fidelity(state, space) == pytest.approx(1.0, abs=1e-12)
    rng = np.random.default_rng(0)
    random = rng.normal(size=state.size) + 1j * rng.normal(size=state.size)
    assert 0 <= ground_state_fidelity(random / np.linalg.norm(random), space) <= 1
