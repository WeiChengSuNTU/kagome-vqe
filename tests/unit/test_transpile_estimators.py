import numpy as np
import pytest

from kagome_vqe.ansatz import HVAnsatz
from kagome_vqe.backends import fake_backend_names, resolve_fake_backend
from kagome_vqe.config import BackendConfig
from kagome_vqe.estimators import (
    AerEnergy,
    EnergyEstimator,
    RuntimeEnergy,
    StatevectorEnergy,
    make_estimator,
)
from kagome_vqe.lattice import GUADALUPE_HEAVY_HEX_LOOP
from kagome_vqe.transpile import compile_for_backend


@pytest.fixture(scope="module")
def guadalupe():
    return resolve_fake_backend("FakeGuadalupeV2")


def test_fake_backend_discovery() -> None:
    names = fake_backend_names()
    assert "FakeGuadalupeV2" in names
    assert resolve_fake_backend("fake_guadalupe").num_qubits == 16
    with pytest.raises(KeyError, match="unknown fake backend"):
        resolve_fake_backend("FakeNowhere")


def test_logical_problem_is_decomposed(hamiltonian) -> None:
    problem = compile_for_backend(HVAnsatz().build(), hamiltonian, None)
    assert problem.backend_name == "logical"
    assert problem.physical_qubits is None
    assert "eswap" not in problem.gate_counts
    assert problem.two_qubit_gates > 0


def test_apply_layout_keeps_energy(hamiltonian, guadalupe) -> None:
    """The transpiled circuit + laid-out observable give the same energy as the logical pair."""
    ansatz = HVAnsatz()
    circuit = ansatz.build()
    logical = compile_for_backend(circuit, hamiltonian, None)
    isa = compile_for_backend(
        circuit,
        hamiltonian,
        guadalupe,
        optimization_level=3,
        initial_layout=GUADALUPE_HEAVY_HEX_LOOP,
        seed_transpiler=1,
    )
    assert isa.circuit.num_qubits == 16
    assert isa.observable.num_qubits == 16
    assert isa.physical_qubits == GUADALUPE_HEAVY_HEX_LOOP
    assert [p.name for p in isa.circuit.parameters] == [p.name for p in circuit.parameters]
    assert set(isa.gate_counts) <= {"rz", "sx", "x", "cx", "id", "delay", "measure", "reset"}

    params = np.array([[0.0, 0.0], [0.3, 0.1], [0.7, 0.4]])
    with StatevectorEnergy(logical) as a, StatevectorEnergy(isa) as b:
        assert a.evaluate(params).energies == pytest.approx(b.evaluate(params).energies, abs=1e-9)
        assert a.evaluate(params).energies[0] == pytest.approx(-18.0)


def test_transpile_without_initial_layout_also_works(hamiltonian, guadalupe) -> None:
    isa = compile_for_backend(HVAnsatz().build(), hamiltonian, guadalupe, seed_transpiler=2)
    assert isa.physical_qubits is not None
    assert len(set(isa.physical_qubits)) == 12
    with StatevectorEnergy(isa) as est:
        assert est.evaluate(np.zeros((1, 2))).energies[0] == pytest.approx(-18.0)


def test_statevector_estimator_batches(hamiltonian) -> None:
    problem = compile_for_backend(HVAnsatz().build(), hamiltonian, None)
    with StatevectorEnergy(problem, seed=0) as est:
        assert isinstance(est, EnergyEstimator)
        batch = est.evaluate(np.zeros((5, 2)))
        assert batch.energies.shape == (5,)
        assert batch.stds.shape == (5,)
        assert np.all(batch.stds == 0)
        single = est.evaluate(np.zeros(2))
        assert single.energies.shape == (1,)
        with pytest.raises(ValueError, match="shape"):
            est.evaluate(np.zeros((1, 1, 2)))


def test_aer_ideal_is_shot_noisy_but_close(hamiltonian) -> None:
    problem = compile_for_backend(HVAnsatz().build(), hamiltonian, None)
    with AerEnergy(problem, shots=2000, seed=1) as est:
        batch = est.evaluate(np.zeros((1, 2)))
    assert batch.shots == 2000
    assert batch.energies[0] == pytest.approx(-18.0, abs=0.5)


def test_make_estimator_dispatch(hamiltonian, guadalupe) -> None:
    problem = compile_for_backend(HVAnsatz().build(), hamiltonian, None)
    assert isinstance(
        make_estimator(BackendConfig(kind="statevector"), problem, None), StatevectorEnergy
    )
    assert isinstance(make_estimator(BackendConfig(kind="aer_ideal"), problem, None), AerEnergy)
    noisy = make_estimator(
        BackendConfig(kind="aer_noisy", name="FakeGuadalupeV2"), problem, guadalupe
    )
    assert noisy.description["noise_backend"] == "fake_guadalupe"
    with pytest.raises(ValueError, match="needs a backend"):
        make_estimator(BackendConfig(kind="fake", name="FakeGuadalupeV2"), problem, None)


def test_runtime_estimator_requires_context(hamiltonian, guadalupe) -> None:
    problem = compile_for_backend(HVAnsatz().build(), hamiltonian, guadalupe, seed_transpiler=1)
    est = RuntimeEnergy(problem, guadalupe, BackendConfig(kind="fake", name="FakeGuadalupeV2"))
    with pytest.raises(RuntimeError, match="context manager"):
        est.evaluate(np.zeros((1, 2)))
