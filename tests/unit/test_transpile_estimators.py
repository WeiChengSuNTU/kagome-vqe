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


def _spread(problem, shots: int, rows: int = 40) -> tuple[float, float, np.ndarray]:
    """Mean and empirical spread of the energy over ``rows`` identical parameter rows."""
    x = np.full((rows, 2), 0.3)
    with AerEnergy(problem, shots=shots, seed=3) as est:
        batch = est.evaluate(x)
    return float(batch.energies.mean()), float(batch.energies.std(ddof=1)), batch.stds


def test_aer_samples_real_shot_noise(hamiltonian) -> None:
    """Regression: Aer's own estimator ignored shots; energies must now be sampled."""
    problem = compile_for_backend(HVAnsatz().build(), hamiltonian, None)
    with StatevectorEnergy(problem) as exact_est:
        exact = float(exact_est.evaluate(np.full((1, 2), 0.3)).energies[0])
    mean_lo, spread_lo, stds_lo = _spread(problem, shots=256)
    mean_hi, spread_hi, _ = _spread(problem, shots=4096)
    assert spread_lo > 0
    assert spread_lo / spread_hi > 2.5  # sqrt(4096 / 256) = 4 in expectation
    assert np.all(stds_lo >= spread_lo)  # the primitive's std is a conservative bound
    assert mean_hi == pytest.approx(exact, abs=5 * spread_hi / np.sqrt(40) + 1e-9)
    assert mean_lo == pytest.approx(exact, abs=5 * spread_lo / np.sqrt(40) + 1e-9)


def test_fresh_seed_per_call_but_reproducible(hamiltonian) -> None:
    """Regression: a fixed seed gave each batch row the same noise on every call."""
    problem = compile_for_backend(HVAnsatz().build(), hamiltonian, None)
    x = np.full((3, 2), 0.3)
    with AerEnergy(problem, shots=512, seed=9) as a, AerEnergy(problem, shots=512, seed=9) as b:
        a1, a2 = a.evaluate(x).energies, a.evaluate(x).energies
        b1, b2 = b.evaluate(x).energies, b.evaluate(x).energies
    assert not np.allclose(a1, a2)  # successive calls are independent
    assert np.allclose(a1, b1)  # but the same run seed reproduces the sequence
    assert np.allclose(a2, b2)


def test_shot_override_and_method_choice(hamiltonian, guadalupe) -> None:
    problem = compile_for_backend(HVAnsatz().build(), hamiltonian, None)
    with AerEnergy(problem, shots=128, seed=1) as est:
        assert est.description["method"] == "statevector"
        assert est.evaluate(np.zeros((1, 2))).shots == 128
        assert est.evaluate(np.zeros((1, 2)), shots=1024).shots == 1024
    isa = compile_for_backend(
        HVAnsatz().build(), hamiltonian, guadalupe, initial_layout=GUADALUPE_HEAVY_HEX_LOOP
    )
    noisy = AerEnergy(isa, shots=128, noise_backend=guadalupe)
    assert noisy.description["method"] == "density_matrix"
    assert isa.measurement_groups == 3
    with pytest.raises(ValueError, match="shots"):
        AerEnergy(problem, shots=0)


@pytest.mark.fake
def test_readout_error_now_affects_energy(hamiltonian, guadalupe) -> None:
    """Regression: readout error was never applied because nothing was measured."""
    from qiskit_aer.noise import NoiseModel

    isa = compile_for_backend(
        HVAnsatz().build(), hamiltonian, guadalupe, initial_layout=GUADALUPE_HEAVY_HEX_LOOP
    )
    readout_only = NoiseModel.from_backend(guadalupe, gate_error=False, thermal_relaxation=False)
    with AerEnergy(isa, shots=4096, noise_model=readout_only, seed=2, method="statevector") as est:
        energy = float(est.evaluate(np.zeros((1, 2))).energies[0])
    assert energy > -17.5  # exact is -18; readout error alone costs most of an energy unit


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
