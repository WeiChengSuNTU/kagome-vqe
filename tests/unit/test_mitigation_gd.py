"""Error mitigation on PRB 106 layer noise: exchange shift rule, ZNE, mitigated GD, trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from kagome_vqe.ansatz import GraphHVAnsatz
from kagome_vqe.ansatz.unshare import EXCHANGE_SHIFT, ParameterMap, unshare_parameters
from kagome_vqe.config import LatticeConfig, OptimizerConfig, RunConfig, ZneConfig
from kagome_vqe.hamiltonian import heisenberg_hamiltonian
from kagome_vqe.mitigation import extrapolate, extrapolation_coefficients, zne_gradient_weights
from kagome_vqe.noise import (
    PauliChannel,
    PauliTrajectorySimulator,
    exact_noisy_state,
    fold_noise_layers,
    layer_noise_circuit,
    measurement_bases,
    noise_locations,
    scale_noise_probability,
)
from kagome_vqe.optimizers import ParameterShift
from kagome_vqe.results import load_run, run_and_save
from kagome_vqe.study import StudyConfig, fitted_maxiter
from kagome_vqe.vqe import prepare

RING4 = {"kind": "custom", "num_sites": 4, "bonds": [[0, 1], [1, 2], [2, 3], [3, 0]]}
RECT6 = {"kind": "kagome_bm_rectangle", "rows": 2, "cols": 3}


def _ansatz(lattice: dict[str, object] = RING4, **kwargs: Any) -> GraphHVAnsatz:
    return GraphHVAnsatz(LatticeConfig.model_validate(lattice).build(), **kwargs)


def _energy_fn(lattice: dict[str, object], circuit: Any) -> Any:
    h = heisenberg_hamiltonian(LatticeConfig.model_validate(lattice).build())

    def energy(phi: np.ndarray) -> float:
        state = Statevector(circuit.assign_parameters(phi))
        return float(np.real(state.expectation_value(h)))

    return energy


def _gd_config(**extra: Any) -> RunConfig:
    base: dict[str, Any] = {
        "name": "zne-gd-test",
        "seed": 3,
        "x0_scale": 0.3,
        "lattice": RING4,
        "ansatz": {"type": "graph_hva", "reps": 1},
        "backend": {
            "kind": "logical_noise",
            "shots": 256,
            "final_shots": 512,
            "logical_noise": {"channel": "depolarizing", "probability": 0.01, "fast_kernels": True},
        },
        "optimizer": {
            "type": "gd",
            "gradient": "parameter_shift",
            "lr": 0.05,
            "maxiter": 2,
            "measure_current": True,
        },
    }
    base.update(extra)
    return RunConfig.model_validate(base)


# -- exchange-gate shift rule ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lattice", "kwargs"),
    [
        (RING4, {"reps": 2}),
        (RECT6, {"reps": 1, "include_h0": True}),
        (RING4, {"reps": 2, "parameterization": "per_layer"}),
    ],
)
def test_exchange_shift_rule_is_exact(lattice: dict[str, object], kwargs: Any) -> None:
    ansatz = _ansatz(lattice, **kwargs)
    circuit, mapping = unshare_parameters(ansatz.build(), exchange_gates=True)
    gates = sum(1 for g in ansatz.gates if not g.pruned)
    assert mapping.num_occurrences == gates  # one occurrence per exchange gate
    assert np.allclose(mapping.occurrence_shifts, EXCHANGE_SHIFT)
    energy = _energy_fn(lattice, circuit)
    theta = np.random.default_rng(1).uniform(-1, 1, ansatz.num_parameters)
    estimate = ParameterShift(mapping).estimate(_ExpandedObjective(energy, mapping), theta)
    h = 1e-5
    fd = np.empty(theta.size)
    for i in range(theta.size):
        step = np.zeros(theta.size)
        step[i] = h
        fd[i] = (energy(mapping.expand(theta + step)) - energy(mapping.expand(theta - step))) / (
            2 * h
        )
    assert np.allclose(estimate.grad, fd, atol=1e-6)
    assert estimate.nfev == 2 * gates


def test_default_unsharing_is_unchanged() -> None:
    circuit, mapping = unshare_parameters(_ansatz(RING4).build())
    assert mapping.shifts is None
    assert mapping.coefficients is None
    assert mapping.num_occurrences == 3 * 4  # rzz, ryy, rxx per bond
    assert circuit.num_parameters == 12


def test_parameter_shift_include_value_reports_the_current_energy() -> None:
    ansatz = _ansatz(RING4)
    circuit, mapping = unshare_parameters(ansatz.build(), exchange_gates=True)
    energy = _energy_fn(RING4, circuit)
    theta = np.array([0.1, 0.4, -0.2, 0.3])
    est = ParameterShift(mapping, include_value=True).estimate(
        _ExpandedObjective(energy, mapping), theta
    )
    assert est.value == pytest.approx(energy(mapping.expand(theta)))
    assert est.nfev == 2 * 4 + 1


def test_shift_rule_is_exact_under_layer_noise() -> None:
    ansatz = _ansatz(RING4)
    marked, mapping = unshare_parameters(
        ansatz.build(mark_noise_locations=True), exchange_gates=True
    )
    from kagome_vqe.noise import insert_layer_noise

    noisy = insert_layer_noise(marked, "depolarizing", 0.05)
    h = heisenberg_hamiltonian(LatticeConfig.model_validate(RING4).build()).to_matrix()

    def energy(phi: np.ndarray) -> float:
        return float(np.real(np.trace(h @ exact_noisy_state(noisy, phi))))

    theta = np.array([0.3, -0.1, 0.5, 0.2])
    est = ParameterShift(mapping).estimate(_ExpandedObjective(energy, mapping), theta)
    eps = 1e-4
    fd = [
        (energy(mapping.expand(theta + eps * e)) - energy(mapping.expand(theta - eps * e)))
        / (2 * eps)
        for e in np.eye(4)
    ]
    assert np.allclose(est.grad, fd, atol=1e-5)


class _ExpandedObjective:
    """A minimal ExpandableObjective around an energy of circuit parameters."""

    def __init__(self, energy: Any, mapping: ParameterMap) -> None:
        self._energy = energy
        self._map = mapping
        self.last_scales: dict[str, Any] | None = None

    @property
    def parameter_map(self) -> ParameterMap:
        return self._map

    def __call__(self, x: np.ndarray) -> float:
        return float(self._energy(self._map.expand(x)))

    def batch(self, xs: np.ndarray) -> np.ndarray:
        return np.array([self(x) for x in np.atleast_2d(xs)])

    def batch_expanded(self, phis: np.ndarray, *, base: np.ndarray | None = None) -> np.ndarray:
        del base
        return np.array([self._energy(phi) for phi in np.atleast_2d(phis)])


# -- noise amplification --------------------------------------------------------------------------


def test_probability_scaling_multiplies_every_channel() -> None:
    circuit = layer_noise_circuit(_ansatz(), "depolarizing", 0.01)
    scaled = scale_noise_probability(circuit, 2.5)
    before = [
        i.operation.probabilities for i in circuit.data if isinstance(i.operation, PauliChannel)
    ]
    after = [
        i.operation.probabilities for i in scaled.data if isinstance(i.operation, PauliChannel)
    ]
    assert len(before) == len(after) == noise_locations(circuit)
    assert np.allclose(np.array(after), 2.5 * np.array(before))
    assert scaled.count_ops().get("rzz") == circuit.count_ops().get("rzz")
    with pytest.raises(ValueError, match="exceed 1"):
        scale_noise_probability(layer_noise_circuit(_ansatz(), "bit_flip", 0.6), 2.0)


@pytest.mark.parametrize("folds", [1, 2])
def test_layer_folding_multiplies_locations_and_keeps_the_unitary(folds: int) -> None:
    ansatz = _ansatz(RING4, reps=2)
    noisy = layer_noise_circuit(ansatz, "depolarizing", 0.0)
    folded = fold_noise_layers(noisy, folds)
    assert noise_locations(folded) == (2 * folds + 1) * noise_locations(noisy)
    theta = np.random.default_rng(0).uniform(-1, 1, ansatz.num_parameters)
    a = PauliTrajectorySimulator(noisy).ideal_state(theta)
    b = PauliTrajectorySimulator(folded).ideal_state(theta)
    assert abs(abs(np.vdot(a, b)) - 1) < 1e-10
    assert set(folded.parameters) == set(noisy.parameters)


def test_folding_and_probability_scaling_agree_to_first_order() -> None:
    ansatz = _ansatz(RING4)
    theta = np.array([0.3, -0.2, 0.4, 0.1])
    h = heisenberg_hamiltonian(LatticeConfig.model_validate(RING4).build()).to_matrix()
    base = layer_noise_circuit(ansatz, "depolarizing", 0.002)

    def energy(circuit: Any) -> float:
        return float(np.real(np.trace(h @ exact_noisy_state(circuit, theta))))

    e1, e_prob, e_fold = (
        energy(base),
        energy(scale_noise_probability(base, 3)),
        energy(fold_noise_layers(base, 1)),
    )
    assert e_prob > e1
    assert e_fold > e1
    # Both amplify the bias ~3x at small p (they differ at second order).
    assert (e_fold - e1) == pytest.approx(e_prob - e1, rel=0.05)


# -- extrapolation weights ------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["linear", "quadratic", "richardson"])
def test_polynomial_weights_reproduce_extrapolate(method: str) -> None:
    scales = [1.0, 2.0, 3.0]
    values = np.array([-17.3, -16.8, -16.3])
    stds = np.array([0.1, 0.12, 0.13])
    c = extrapolation_coefficients(scales, stds, method)  # type: ignore[arg-type]
    assert float(c @ values) == pytest.approx(extrapolate(scales, values, stds, method)[0])  # type: ignore[arg-type]
    w = zne_gradient_weights(scales, values, stds, method)  # type: ignore[arg-type]
    assert np.allclose(w, c)


def test_exponential_weights_are_the_chain_rule() -> None:
    scales = [1.0, 2.0, 3.0]
    values = np.array([-17.3, -16.8, -16.3])
    stds = np.array([0.1, 0.12, 0.13])
    w = zne_gradient_weights(scales, values, stds, "exponential")
    eps = 1e-6
    for i in range(3):
        bumped = values.copy()
        bumped[i] += eps
        # stds held fixed relative to the original values (the weights are frozen)
        up = _exp_fixed(scales, bumped, stds, values)
        bumped[i] -= 2 * eps
        down = _exp_fixed(scales, bumped, stds, values)
        assert w[i] == pytest.approx((up - down) / (2 * eps), rel=1e-5)


def _exp_fixed(scales: list[float], y: np.ndarray, s: np.ndarray, y0: np.ndarray) -> float:
    c = extrapolation_coefficients(scales, s / np.abs(y0), "exponential")
    return float(np.sign(y[0]) * np.exp(c @ np.log(np.abs(y))))


def test_zne_parameter_shift_combines_scale_gradients_with_the_weights() -> None:
    """With per-scale energies E_l(phi) = a_l * E(phi) + b_l the result is d E_ZNE / d theta."""
    ansatz = _ansatz(RING4)
    circuit, mapping = unshare_parameters(ansatz.build(), exchange_gates=True)
    energy = _energy_fn(RING4, circuit)
    damping = np.array([0.95, 0.90, 0.85])
    offset = np.array([0.1, 0.2, 0.3])

    class Scaled(_ExpandedObjective):
        def batch_expanded(self, phis: np.ndarray, *, base: np.ndarray | None = None) -> np.ndarray:
            clean = np.array([self._energy(phi) for phi in phis])
            per_scale = damping[:, None] * clean[None, :] + offset[:, None]
            self.last_scales = {
                "scales": [1.0, 2.0, 3.0],
                "energies": per_scale,
                "stds": np.full_like(per_scale, 0.1),
                "extrapolation": "linear",
            }
            return np.array(
                [extrapolate([1, 2, 3], col, np.full(3, 0.1), "linear")[0] for col in per_scale.T]
            )

    theta = np.array([0.2, -0.3, 0.1, 0.4])
    est = ParameterShift(mapping, include_value=True).estimate(Scaled(energy, mapping), theta)
    exact = ParameterShift(mapping).estimate(_ExpandedObjective(energy, mapping), theta).grad
    c = extrapolation_coefficients([1, 2, 3], np.full(3, 0.1), "linear")
    assert np.allclose(est.grad, (c @ damping) * exact)
    assert est.details["scale_grads"].shape == (3, 4)
    assert np.allclose(est.details["weights"], c)
    assert est.value == pytest.approx(c @ (damping * energy(mapping.expand(theta)) + offset))


# -- configuration --------------------------------------------------------------------------------


def test_zne_config_scales() -> None:
    assert ZneConfig().scales == [1, 3, 5]
    assert ZneConfig(scales=[1, 1.5, 2], noise_scaling="probability").scales == [1, 1.5, 2]
    with pytest.raises(ValueError, match="odd integers"):
        ZneConfig(scales=[1, 2, 3])
    with pytest.raises(ValueError, match="start with 1"):
        ZneConfig(scales=[3, 5])
    with pytest.raises(ValueError, match="distinct"):
        ZneConfig(scales=[1, 1], noise_scaling="probability")
    assert "probability" in ZneConfig(scales=[1, 2], noise_scaling="probability").label()


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"mitigation": {"zne": {"scales": [1, 3]}}}, "probability' or 'layer_folding"),
        (
            {
                "mitigation": {"symmetry": True},
                "backend": {
                    "kind": "logical_noise",
                    "shots": 8,
                    "logical_noise": {"engine": "aer"},
                },
            },
            "trajectories",
        ),
        (
            {
                "mitigation": {"zne": {"scales": [1, 2], "noise_scaling": "probability"}},
                "optimizer": {"type": "gd", "gradient": "parameter_shift"},
            },
            "measure_current",
        ),
        ({"track_ideal_gradient": True}, "measure_current"),
    ],
)
def test_logical_noise_mitigation_validation(update: dict[str, Any], message: str) -> None:
    base = _gd_config().model_dump(mode="json")
    base["optimizer"]["measure_current"] = False
    base.update(update)
    with pytest.raises(ValueError, match=message):
        RunConfig.model_validate(base)


def test_layer_noise_scaling_needs_logical_noise() -> None:
    with pytest.raises(ValueError, match=r"needs backend\.kind = 'logical_noise'"):
        RunConfig.model_validate(
            {
                "backend": {"kind": "aer_ideal", "shots": 100},
                "mitigation": {"zne": {"scales": [1, 2], "noise_scaling": "probability"}},
            }
        )


def test_measure_current_needs_parameter_shift() -> None:
    with pytest.raises(ValueError, match="measure_current"):
        OptimizerConfig(type="spsa", measure_current=True)
    assert (
        "current point"
        in OptimizerConfig(
            type="gd", gradient="parameter_shift", measure_current=True
        ).gradient_description()
    )


# -- end to end -----------------------------------------------------------------------------------


def test_noisy_parameter_shift_gd_records_a_trajectory(tmp_path: Path) -> None:
    cfg = _gd_config(track_ideal_gradient=True)
    prep = prepare(cfg)
    assert prep.parameter_map.num_occurrences == 4
    run = load_run(run_and_save(cfg, root=tmp_path))
    traj = run.trajectory
    assert traj is not None
    assert traj["theta"].shape == (2, 4)
    assert np.allclose(traj["theta"][0], run.x0)
    assert np.allclose(traj["theta_next"], traj["theta"] - 0.05 * traj["grad"])
    assert traj["scales"].tolist() == [1.0]
    assert np.allclose(traj["scale_values"][:, 0], traj["value"])
    assert traj["evaluations"].tolist() == [9, 18]
    groups = 3
    assert traj["circuit_executions"].tolist() == [9 * groups, 18 * groups]
    assert traj["shots"].tolist() == [9 * 256 * groups, 18 * 256 * groups]
    assert np.all(np.isfinite(traj["ideal_grad"]))
    assert np.all(np.isfinite(traj["ideal_energy"]))
    assert (
        run.resources["circuit_executions"] == (18 + 1 + 1) * groups
    )  # + GD final + re-measurement


@pytest.mark.parametrize("scaling", ["probability", "layer_folding"])
def test_zne_gd_records_scale_resolved_gradients(tmp_path: Path, scaling: str) -> None:
    scales = [1, 2, 3] if scaling == "probability" else [1, 3]
    cfg = _gd_config(
        mitigation={"zne": {"scales": scales, "extrapolation": "linear", "noise_scaling": scaling}}
    )
    run = load_run(run_and_save(cfg, root=tmp_path))
    traj = run.trajectory
    assert traj is not None
    L = len(scales)
    assert traj["scale_values"].shape == (2, L)
    assert traj["scale_grads"].shape == (2, L, 4)
    for t in range(2):
        assert np.allclose(traj["grad"][t], traj["zne_weights"][t] @ traj["scale_grads"][t])
    assert run.resources["circuit_executions"] == (18 + 1 + 1) * 3 * L
    estimator = run.result["estimator"]
    assert estimator["kind"] == "zne"
    assert estimator["noise_scaling"] == scaling
    assert run.metadata["final_zne"]["scales"] == [float(s) for s in scales]
    assert run.metadata["final_zne"]["unmitigated_energy"] == pytest.approx(
        run.metadata["final_zne"]["energies"][0]
    )


def test_zne_reduces_the_bias_of_a_fixed_point_in_expectation() -> None:
    """Exact energies at scaled noise, linear extrapolation: closer to the noise-free value."""
    ansatz = _ansatz(RING4)
    theta = np.array([0.3, -0.2, 0.4, 0.1])
    h = heisenberg_hamiltonian(LatticeConfig.model_validate(RING4).build()).to_matrix()
    base = layer_noise_circuit(ansatz, "depolarizing", 0.01)
    values = np.array(
        [
            float(np.real(np.trace(h @ exact_noisy_state(scale_noise_probability(base, s), theta))))
            for s in (1, 2, 3)
        ]
    )
    clean = PauliTrajectorySimulator(layer_noise_circuit(ansatz, "depolarizing", 0.0))
    psi = clean.ideal_state(theta)
    ideal = float(np.real(np.vdot(psi, h @ psi)))
    zne, _ = extrapolate([1, 2, 3], values, np.ones(3), "richardson")
    assert abs(zne - ideal) < 0.1 * abs(values[0] - ideal)


def test_fast_kernels_sample_the_same_distribution() -> None:
    ansatz = _ansatz(RECT6)
    circuit = layer_noise_circuit(ansatz, "depolarizing", 0.02)
    theta = np.random.default_rng(2).uniform(-1, 1, ansatz.num_parameters)
    ref, fast = PauliTrajectorySimulator(circuit), PauliTrajectorySimulator(circuit, fast=True)
    assert np.allclose(ref.ideal_state(theta), fast.ideal_state(theta))
    h = heisenberg_hamiltonian(LatticeConfig.model_validate(RECT6).build())
    bases = measurement_bases(h)
    rng = np.random.default_rng(5)
    energies = []
    for sim in (ref, fast):
        total = 0.0
        for _ in range(4):
            histos = sim.sample_measurements(theta, [b for b, _ in bases], 4096, rng)
            total += sum(float(c @ d) / 4096 for c, (_, d) in zip(histos, bases, strict=True))
        energies.append(total / 4)
    exact = float(np.real(np.trace(h.to_matrix() @ exact_noisy_state(circuit, theta))))
    assert energies[0] == pytest.approx(exact, abs=0.08)
    assert energies[1] == pytest.approx(exact, abs=0.08)
    from kagome_vqe.noise import _basis_probabilities, _basis_probabilities_reference

    states = np.vstack([ref.ideal_state(theta), fast.ideal_state(theta + 0.1)])
    for basis in ("XXXXXX", "YYYYYY", "ZXYZXY"):
        assert np.allclose(
            _basis_probabilities(states, basis, 6), _basis_probabilities_reference(states, basis, 6)
        )


# -- studies --------------------------------------------------------------------------------------


def test_study_backend_mitigation_and_parameter_shift_budget() -> None:
    study = StudyConfig.model_validate(
        {
            "name": "zne_study",
            "seeds": [1],
            "track_ideal_gradient": True,
            "lattices": [{"label": "ring", "lattice": RING4}],
            "backends": [
                {
                    "label": "dep_zne",
                    "backend": _gd_config().backend.model_dump(mode="json"),
                    "budget": {"max_shots": 256 * 3 * 3 * 37},
                    "mitigation": {
                        "zne": {
                            "scales": [1, 2, 3],
                            "extrapolation": "linear",
                            "noise_scaling": "probability",
                        }
                    },
                }
            ],
            "optimizers": [
                {
                    "label": "gd_ps",
                    "fit_maxiter_to_budget": True,
                    "optimizer": {
                        "type": "gd",
                        "gradient": "parameter_shift",
                        "lr": 0.05,
                        "measure_current": True,
                    },
                }
            ],
        }
    )
    (run,) = study.runs()
    assert run.config.mitigation.zne is not None
    assert run.config.track_ideal_gradient is True
    # 37 evaluations at 3 scales x 3 bases x 256 shots; 9 per iteration + 1 final -> 4
    assert run.config.optimizer.maxiter == 4
    assert (
        fitted_maxiter(
            run.config.optimizer, study.backends[0], study.lattices[0].lattice, 4, num_occurrences=4
        )
        == 4
    )


# -- fixed-parameter ZNE analysis -----------------------------------------------------------------


def test_zne_analysis_on_saved_parameters(tmp_path: Path) -> None:
    from kagome_vqe.zne_analysis import ZneAnalysisConfig, run_zne_analysis

    source = RunConfig.model_validate(
        {
            "name": "noiseless-src",
            "lattice": RING4,
            "ansatz": {"type": "graph_hva", "reps": 1},
            "backend": {"kind": "statevector"},
            "optimizer": {"type": "scipy", "method": "L-BFGS-B", "maxiter": 50},
        }
    )
    run_dir = run_and_save(source, root=tmp_path / "runs")
    cfg = ZneAnalysisConfig(
        name="zne-test",
        runs=[str(run_dir)],
        probabilities=[0.005, 0.02],
        methods=[
            {
                "label": "prob_lin",
                "noise_scaling": "probability",
                "scales": [1, 2, 3],
                "extrapolation": "linear",
            },
            {
                "label": "fold_rich",
                "noise_scaling": "layer_folding",
                "scales": [1, 3, 5],
                "extrapolation": "richardson",
            },
        ],
        shots=512,
        sampled_repeats=3,
    )
    assert len(cfg.levels()) == 5  # lambda = 1 shared
    out = run_zne_analysis(cfg, output_root=tmp_path / "analysis")
    import pandas as pd

    levels = pd.read_csv(out / "levels.csv")
    points = pd.read_csv(out / "points.csv")
    assert len(levels) == 2 * 5
    assert len(points) == 2 * 2
    assert (levels["bias"] > 0).all()
    assert (points["zne_bias"].abs() < points["unmitigated_bias"].abs()).all()
    assert (points["zne_se"] > points["unmitigated_se"]).all()
    assert (points["layer_cost_per_energy"] == [3, 9, 3, 9]).all()
    assert (out / "sampled.csv").exists()
    assert (out / "report.md").read_text().count("| 1 | depolarizing |") == 4
    assert any((out / "figures").glob("zne_bias_*.png"))


def test_study_summary_includes_gradient_trajectories(tmp_path: Path) -> None:
    from kagome_vqe.study import run_study

    backend = _gd_config().backend.model_dump(mode="json")
    study = StudyConfig.model_validate(
        {
            "name": "traj_study",
            "seeds": [1, 2],
            "output_dir": str(tmp_path),
            "track_ideal_gradient": True,
            "lattices": [{"label": "ring", "lattice": RING4}],
            "backends": [
                {"label": "noisy", "backend": backend},
                {
                    "label": "zne",
                    "backend": backend,
                    "mitigation": {
                        "zne": {
                            "scales": [1, 2],
                            "extrapolation": "linear",
                            "noise_scaling": "probability",
                        }
                    },
                },
            ],
            "optimizers": [
                {
                    "label": "gd_ps",
                    "optimizer": {
                        "type": "gd",
                        "gradient": "parameter_shift",
                        "lr": 0.05,
                        "maxiter": 2,
                        "measure_current": True,
                    },
                }
            ],
        }
    )
    study_dir = run_study(study)
    summary = study_dir / "summary"
    text = (summary / "trajectories.md").read_text()
    assert "| 1 | noisy | gd_ps | 2 |" in text
    assert "| 1 | zne | gd_ps | 2 |" in text
    import pandas as pd

    per_run = pd.read_csv(summary / "trajectory_runs.csv")
    assert len(per_run) == 4
    assert per_run["mean_cosine_to_ideal"].between(-1, 1).all()
    assert (summary / "trajectories.png").exists()
    runs = pd.read_csv(summary / "runs.csv")
    assert set(runs["mitigation"]) == {"none", "zne"}
