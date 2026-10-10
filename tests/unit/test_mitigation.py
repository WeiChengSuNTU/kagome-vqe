"""Roadmap M2: counts-level mitigation, ZNE, twin rescaling and fixed-point evaluation."""

from pathlib import Path
from types import TracebackType
from typing import Any, Self

import numpy as np
import pytest
from pydantic import ValidationError
from qiskit.quantum_info import Operator, SparsePauliOp
from typer.testing import CliRunner

from kagome_vqe.ansatz import HVAnsatz
from kagome_vqe.backends import resolve_fake_backend
from kagome_vqe.cli import app
from kagome_vqe.config import (
    AnsatzConfig,
    BackendConfig,
    MitigationConfig,
    RescaleConfig,
    RunConfig,
    ZneConfig,
)
from kagome_vqe.estimators import EnergyBatch, StatevectorEnergy, make_estimator
from kagome_vqe.lattice import GUADALUPE_HEAVY_HEX_LOOP
from kagome_vqe.mitigation import (
    CountsEnergy,
    ReadoutCalibration,
    TwinRescaledEnergy,
    ZneEnergy,
    choose_twin,
    extrapolate,
    measurement_plan,
)
from kagome_vqe.transpile import IsaProblem, compile_for_backend, fold_two_qubit_gates
from kagome_vqe.vqe import evaluate_fixed

IDEAL = BackendConfig(kind="aer_ideal", shots=4096)


class Damped:
    """Fake estimator: energy -18 damped by 0.99 per two-qubit gate of its problem."""

    def __init__(self, problem: IsaProblem, scale: float = 1.0) -> None:
        self.factor = 0.99 ** (problem.two_qubit_gates * scale)
        self.description: dict[str, Any] = {"kind": "damped"}

    def evaluate(self, params: np.ndarray, *, shots: int | None = None) -> EnergyBatch:
        del shots
        k = np.atleast_2d(params).shape[0]
        return EnergyBatch(energies=np.full(k, -18.0 * self.factor), stds=np.full(k, 1e-3))

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


# -- counts-level --------------------------------------------------------------------------------


def test_measurement_plan_of_heisenberg(hamiltonian: SparsePauliOp) -> None:
    groups, constant = measurement_plan(hamiltonian)
    assert constant == 0.0
    assert sorted(set(g.basis) for g in groups) == [{"X"}, {"Y"}, {"Z"}]
    assert all(len(g.supports) == 18 for g in groups)
    assert all(len(s) == 2 for g in groups for s in g.supports)


def test_readout_weights_are_unbiased(rng: np.random.Generator) -> None:
    """Independent flips + per-shot weights recover the noise-free two-qubit parity."""
    e0, e1 = np.array([0.08, 0.10]), np.array([0.12, 0.06])
    weights = ReadoutCalibration(e0, e1).weights()
    assert ReadoutCalibration(np.zeros(2), np.zeros(2)).weights() == pytest.approx(
        np.array([[1.0, -1.0], [1.0, -1.0]])
    )
    true_bits = rng.random((400_000, 2)) < np.array([0.3, 0.8])
    flip = np.where(true_bits, rng.random(true_bits.shape) < e1, rng.random(true_bits.shape) < e0)
    noisy = true_bits ^ flip
    truth = np.mean((1 - 2 * true_bits[:, 0]) * (1 - 2 * true_bits[:, 1]))
    raw = np.mean((1 - 2 * noisy[:, 0]) * (1 - 2 * noisy[:, 1]))
    idx = noisy.astype(int)
    corrected = np.mean(weights[0, idx[:, 0]] * weights[1, idx[:, 1]])
    assert abs(raw - truth) > 0.02
    assert corrected == pytest.approx(truth, abs=0.01)
    with pytest.raises(ValueError, match="too noisy"):
        ReadoutCalibration(np.array([0.5]), np.array([0.5])).weights()


def test_counts_energy_matches_statevector(hamiltonian: SparsePauliOp) -> None:
    problem = compile_for_backend(HVAnsatz().build(), hamiltonian, None)
    x = np.array([[0.3, 0.2], [0.0, 0.0]])
    with StatevectorEnergy(problem) as sv:
        exact = sv.evaluate(x).energies
    with CountsEnergy(problem, IDEAL, None, seed=1) as est:
        batch = est.evaluate(x)
    assert batch.energies[0] == pytest.approx(exact[0], abs=5 * batch.stds[0])
    assert batch.stds[0] > 0
    # the singlet product has zero shot noise: every shot gives exactly -6 per basis
    assert batch.energies[1] == pytest.approx(-18.0)
    assert batch.stds[1] == pytest.approx(0.0)
    assert est.evaluate(x, shots=100).shots == 100


def test_symmetry_filter_keeps_every_noise_free_shot(hamiltonian: SparsePauliOp) -> None:
    problem = compile_for_backend(HVAnsatz().build(), hamiltonian, None)
    with CountsEnergy(problem, IDEAL, None, symmetry=True, seed=2) as est:
        est.evaluate(np.array([[0.4, 0.1]]))
        assert est.description["mean_kept_fraction"] == pytest.approx(1.0)


@pytest.mark.fake
def test_readout_and_symmetry_on_noisy_guadalupe(hamiltonian: SparsePauliOp) -> None:
    backend = resolve_fake_backend("FakeGuadalupeV2")
    problem = compile_for_backend(
        HVAnsatz().build(), hamiltonian, backend, initial_layout=GUADALUPE_HEAVY_HEX_LOOP
    )
    cfg = BackendConfig(kind="aer_noisy", name="FakeGuadalupeV2", shots=4096)
    zero = np.zeros((1, 2))
    with CountsEnergy(problem, cfg, backend, seed=3) as raw_est:
        raw = raw_est.evaluate(zero).energies[0]
    with CountsEnergy(
        problem, cfg, backend, readout=True, symmetry=True, readout_shots=8192, seed=3
    ) as est:
        mitigated = est.evaluate(zero).energies[0]
        assert est.calibration is not None
        assert 0 < est.calibration.e0.mean() < 0.1
        assert 0.3 < est.description["mean_kept_fraction"] < 0.8
    assert raw > -13.5
    assert abs(mitigated + 18.0) < abs(raw + 18.0) / 4


# -- folding and ZNE -----------------------------------------------------------------------------


def test_folding_preserves_unitary_and_scales_gates() -> None:
    circuit = HVAnsatz(num_qubits=4).build().decompose(reps=2).assign_parameters([0.3, 0.7])
    folded = fold_two_qubit_gates(circuit, 3)
    assert Operator(folded).equiv(Operator(circuit))
    two_q = sum(1 for i in circuit.data if i.operation.num_qubits == 2)
    assert sum(1 for i in folded.data if i.operation.num_qubits == 2) == 3 * two_q
    with pytest.raises(ValueError, match="odd"):
        fold_two_qubit_gates(circuit, 2)


@pytest.mark.parametrize(
    ("method", "values", "expected"),
    [
        ("linear", [-10.0, -8.0, -6.0], -11.0),
        ("richardson", [-10.0, -8.0, -6.0], -11.0),
        ("quadratic", [-10.0, -8.0, -6.0], -11.0),
        ("exponential", [-18 * 0.9, -18 * 0.9**3, -18 * 0.9**5], -18.0),
    ],
)
def test_extrapolation_models(method: str, values: list[float], expected: float) -> None:
    estimate, err = extrapolate([1, 3, 5], np.array(values), np.full(3, 0.01), method)  # type: ignore[arg-type]
    assert estimate == pytest.approx(expected, rel=1e-9)
    assert err > 0


def test_exponential_falls_back_on_mixed_signs() -> None:
    estimate, _ = extrapolate([1, 3], np.array([-1.0, 1.0]), np.full(2, 0.1), "exponential")
    assert estimate == pytest.approx(-2.0)


def test_zne_recovers_multiplicative_damping(hamiltonian: SparsePauliOp) -> None:
    problem = compile_for_backend(HVAnsatz().build(), hamiltonian, None)
    with ZneEnergy(problem, Damped, (1, 3, 5), "exponential") as zne:
        batch = zne.evaluate(np.zeros((2, 2)))
        assert zne.description["scales"] == [1, 3, 5]
    assert batch.energies == pytest.approx([-18.0, -18.0], rel=1e-9)


# -- twin rescaling -----------------------------------------------------------------------------


def test_twin_rescaling_divides_out_a_uniform_factor(hamiltonian: SparsePauliOp) -> None:
    problem = compile_for_backend(HVAnsatz().build(), hamiltonian, None)
    with TwinRescaledEnergy(Damped(problem), np.zeros(2), -18.0) as est:
        batch = est.evaluate(np.ones((1, 2)))
        assert est.factor == pytest.approx(1 / 0.99**problem.two_qubit_gates)
    assert batch.energies[0] == pytest.approx(-18.0)


def test_choose_twin(hamiltonian: SparsePauliOp, rng: np.random.Generator) -> None:
    twin, energy = choose_twin(HVAnsatz().build(), hamiltonian, -18.0, rng=rng)
    assert twin.shape == (2,)
    assert abs(energy + 18.0) > 1e-6
    assert abs(energy) > 1.0
    # On the Clifford grid every HVA point is a ground state or has zero energy.
    with pytest.raises(ValueError, match="no candidate twin"):
        choose_twin(HVAnsatz().build(), hamiltonian, -18.0, rng=rng, step=np.pi / 2)


# -- configuration and assembly -------------------------------------------------------------------


def test_mitigation_config_validation() -> None:
    noisy = BackendConfig(kind="aer_noisy", name="FakeGuadalupeV2")
    with pytest.raises(ValidationError, match="shot-based"):
        RunConfig(mitigation=MitigationConfig(readout=True))
    with pytest.raises(ValidationError, match="singlet"):
        RunConfig(
            ansatz=AnsatzConfig(type="real_amplitudes"),
            backend=noisy,
            mitigation=MitigationConfig(symmetry=True),
        )
    with pytest.raises(ValidationError, match="odd"):
        ZneConfig(scales=[1, 2])
    full = MitigationConfig(readout=True, symmetry=True, zne=ZneConfig(), rescale=RescaleConfig())
    assert full.label() == "readout+symmetry+zne+twin"
    assert MitigationConfig().label() == "none"


def test_make_estimator_assembles_the_stack(hamiltonian: SparsePauliOp) -> None:
    problem = compile_for_backend(HVAnsatz().build(), hamiltonian, None)
    plain = make_estimator(IDEAL, problem, None, mitigation=MitigationConfig())
    assert plain.description["kind"] == "aer_ideal"
    stack = make_estimator(
        IDEAL,
        problem,
        None,
        mitigation=MitigationConfig(readout=True, zne=ZneConfig(), rescale=RescaleConfig()),
        twin=(np.zeros(2), -18.0),
    )
    assert isinstance(stack, TwinRescaledEnergy)
    assert stack.description["inner"]["kind"] == "zne"
    assert stack.description["inner"]["inner"]["kind"] == "counts:aer_ideal"
    with pytest.raises(ValueError, match="twin"):
        make_estimator(IDEAL, problem, None, mitigation=MitigationConfig(rescale=RescaleConfig()))


def test_evaluate_fixed_and_cli(tmp_path: Path) -> None:
    cfg = RunConfig(
        name="fixed",
        backend=BackendConfig(kind="aer_ideal", shots=256),
        mitigation=MitigationConfig(symmetry=True, readout=True, readout_shots=256),
    )
    result = evaluate_fixed(cfg, 0.3, repeats=3)
    assert result.energies.shape == (3,)
    assert result.ideal_energy == pytest.approx(-16.657, abs=1e-3)
    assert result.spread > 0
    assert result.shots_per_evaluation == 3 * 256
    assert result.setup_shots == 2 * 256
    assert not result.below_ground
    summary = result.summary()
    assert summary["mitigation"] == "readout+symmetry"

    config = tmp_path / "bench.toml"
    config.write_text(
        f'name = "m2t"\noutput_dir = "{tmp_path}"\n'
        '[backend]\nkind = "aer_ideal"\nshots = 128\n'
        '[[combos]]\nansatz = { type = "hva" }\nlabel = "raw"\n'
        '[[combos]]\nansatz = { type = "hva" }\nlabel = "sym"\nmitigation = { symmetry = true }\n'
    )
    out = CliRunner().invoke(app, ["benchmark", str(config), "--point", "0.3", "--repeats", "2"])
    assert out.exit_code == 0, out.output
    table = (tmp_path / "m2t-evaluation-0p3" / "evaluation.md").read_text()
    assert "| m2t-raw | hva | none |" in table
    assert "| m2t-sym | hva | symmetry |" in table
