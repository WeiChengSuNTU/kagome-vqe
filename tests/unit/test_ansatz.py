import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector

from kagome_vqe.ansatz import (
    Ansatz,
    HVAnsatz,
    LibraryAnsatz,
    RVBAnsatz,
    eswap,
    legacy_eswap,
    ring_bonds,
    singlet_pair,
)
from kagome_vqe.hamiltonian import heisenberg_hamiltonian
from kagome_vqe.lattice import Lattice

SWAP = Operator(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))
IDENTITY = Operator(np.eye(4))


def _op(gate) -> Operator:
    qc = QuantumCircuit(2)
    qc.append(gate, [0, 1])
    return Operator(qc)


def test_singlet_pair_energy() -> None:
    qc = QuantumCircuit(2)
    singlet_pair(qc, 0, 1)
    bond = SparsePauliOp(["XX", "YY", "ZZ"])
    assert Statevector(qc).expectation_value(bond).real == pytest.approx(-3.0)


def test_eswap_limits() -> None:
    assert _op(eswap(0.0)).equiv(IDENTITY)
    assert _op(eswap(np.pi)).equiv(SWAP)


@settings(max_examples=20, deadline=None)
@given(st.floats(-2 * np.pi, 2 * np.pi))
def test_eswap_is_exp_of_swap(phi: float) -> None:
    expected = Operator(np.eye(4) * np.cos(phi / 2) - 1j * np.sin(phi / 2) * SWAP.data)
    assert _op(eswap(phi)).equiv(expected)
    assert (_op(eswap(phi)) @ _op(eswap(-phi))).equiv(IDENTITY)


def test_legacy_eswap_carries_an_extra_x() -> None:
    qc = QuantumCircuit(2)
    qc.x(0)
    assert _op(legacy_eswap(0.0)).equiv(Operator(qc))
    assert not _op(legacy_eswap(0.0)).equiv(IDENTITY)


def test_ring_bonds_match_original_ordering() -> None:
    odd, even = ring_bonds(12)
    assert odd == [(11, 0), (9, 10), (7, 8), (5, 6), (3, 4), (1, 2)]
    assert even == [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
    with pytest.raises(ValueError, match="even"):
        ring_bonds(5)


@pytest.mark.parametrize(
    ("ansatz", "expected"),
    [
        (RVBAnsatz(), 12),
        (RVBAnsatz(reps=2), 24),
        (RVBAnsatz(gate="legacy"), 12),
        (HVAnsatz(), 2),
        (HVAnsatz(reps=3), 6),
        (HVAnsatz(extra_bonds=[(0, 2)]), 3),
        (LibraryAnsatz("real_amplitudes", reps=1), 12),
        (LibraryAnsatz("efficient_su2", reps=1), 24),
    ],
)
def test_parameter_counts(ansatz: Ansatz, expected: int) -> None:
    assert isinstance(ansatz, Ansatz)
    circuit = ansatz.build()
    assert ansatz.num_parameters == expected
    assert circuit.num_parameters == expected
    assert circuit.num_qubits == 12


def test_parameter_names_sort_in_creation_order() -> None:
    names = [p.name for p in HVAnsatz(reps=2).build().parameters]
    assert names == ["gamma00", "gamma01", "phi00", "phi01"]
    rvb = [p.name for p in RVBAnsatz(reps=2).build().parameters]
    assert rvb == sorted(rvb)
    assert rvb[0] == "phi00_00"


def test_build_returns_fresh_circuits() -> None:
    ansatz = HVAnsatz()
    assert ansatz.build() is not ansatz.build()


@pytest.mark.parametrize("ansatz", [RVBAnsatz(), HVAnsatz(), RVBAnsatz(gate="legacy")])
def test_zero_parameters_are_the_exact_ground_state(ansatz: Ansatz, lattice: Lattice) -> None:
    hamiltonian = heisenberg_hamiltonian(lattice)
    circuit = ansatz.build().assign_parameters(np.zeros(ansatz.num_parameters))
    assert Statevector(circuit).expectation_value(hamiltonian).real == pytest.approx(-18.0)


@pytest.mark.parametrize("ansatz", [RVBAnsatz(), HVAnsatz()])
def test_identity_order_starts_far_from_ground_state(
    ansatz: Ansatz, identity_lattice: Lattice
) -> None:
    hamiltonian = heisenberg_hamiltonian(identity_lattice)
    circuit = ansatz.build().assign_parameters(np.zeros(ansatz.num_parameters))
    assert Statevector(circuit).expectation_value(hamiltonian).real == pytest.approx(-9.0)


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="reps"):
        HVAnsatz(reps=0)
    with pytest.raises(ValueError, match="eSWAP"):
        RVBAnsatz(gate="nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="extra bond"):
        HVAnsatz(extra_bonds=[(0, 12)])
    with pytest.raises(ValueError, match="library"):
        LibraryAnsatz("nope")  # type: ignore[arg-type]
