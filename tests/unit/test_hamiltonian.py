import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from kagome_vqe.ansatz import singlet_pair
from kagome_vqe.hamiltonian import (
    exact_ground_energy,
    exact_spectrum,
    heisenberg_hamiltonian,
    relative_error,
)
from kagome_vqe.lattice import Lattice, kagome_unit_cell, perfect_matchings

E0 = -18.0
E1 = -16.96132284


def test_term_structure(hamiltonian: SparsePauliOp) -> None:
    assert hamiltonian.num_qubits == 12
    assert len(hamiltonian) == 54
    assert np.allclose(hamiltonian.coeffs, 1.0)
    labels = [str(p) for p in hamiltonian.paulis]
    assert all(lbl.count("I") == 10 for lbl in labels)
    assert all(set(lbl) - {"I"} in ({"X"}, {"Y"}, {"Z"}) for lbl in labels)


def test_hermitian(hamiltonian: SparsePauliOp) -> None:
    dense = hamiltonian.to_matrix()
    assert np.allclose(dense, dense.conj().T)


def test_exact_spectrum(hamiltonian: SparsePauliOp) -> None:
    spectrum = exact_spectrum(hamiltonian, k=3)
    assert spectrum == pytest.approx([E0, E0, E1], abs=1e-6)
    assert exact_ground_energy(hamiltonian) == pytest.approx(E0, abs=1e-6)


def test_dense_fallback_for_tiny_systems() -> None:
    two_site = Lattice(num_sites=2, edges=((0, 1, 1.0),))
    assert exact_spectrum(heisenberg_hamiltonian(two_site), k=4) == pytest.approx([-3, 1, 1, 1])


def test_spectrum_invariant_under_site_order(
    hamiltonian: SparsePauliOp, identity_lattice: Lattice
) -> None:
    other = exact_spectrum(heisenberg_hamiltonian(identity_lattice), k=3)
    assert other == pytest.approx(exact_spectrum(hamiltonian, k=3), abs=1e-6)


def test_dimer_coverings_are_ground_states(lattice: Lattice) -> None:
    """Six singlets on any perfect matching give <H> = -18: the ground state is a dimer product."""
    hamiltonian = heisenberg_hamiltonian(lattice)
    q = lattice.qubit_of_site
    for cover in perfect_matchings(lattice):
        circuit = QuantumCircuit(12)
        for a, b in cover:
            singlet_pair(circuit, q[a], q[b])
        assert Statevector(circuit).expectation_value(hamiltonian).real == pytest.approx(E0)


@settings(max_examples=10, deadline=None)
@given(st.permutations(list(range(12))))
def test_random_site_orders_keep_spectrum(order: list[int]) -> None:
    spectrum = exact_spectrum(heisenberg_hamiltonian(kagome_unit_cell(site_order=order)), k=2)
    assert spectrum == pytest.approx([E0, E0], abs=1e-6)


def test_relative_error() -> None:
    assert relative_error(-18.0, -17.82) == pytest.approx(0.01)
    with pytest.raises(ZeroDivisionError):
        relative_error(0.0, 1.0)
