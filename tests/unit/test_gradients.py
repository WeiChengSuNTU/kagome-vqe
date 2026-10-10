"""Roadmap M1: gradient estimators, parameter shift, the SPSA-ADAM hybrid, ideal tracking."""

from pathlib import Path

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp
from typer.testing import CliRunner

from kagome_vqe.ansatz import Ansatz, HVAnsatz, LibraryAnsatz, RVBAnsatz
from kagome_vqe.ansatz.unshare import ParameterMap, unshare_parameters
from kagome_vqe.cli import app
from kagome_vqe.config import (
    AnsatzConfig,
    BackendConfig,
    BenchmarkConfig,
    ComboConfig,
    OptimizerConfig,
    RunConfig,
)
from kagome_vqe.estimators import StatevectorEnergy
from kagome_vqe.optimizers import (
    Adam,
    CentralDifference,
    ForwardDifference,
    GradientDescent,
    ParameterShift,
    SimultaneousPerturbation,
    Spsa,
    as_objective,
)
from kagome_vqe.plotting import plot_ideal_vs_shots
from kagome_vqe.results import load_run, new_run_dir, save_run, write_benchmark_summary
from kagome_vqe.transpile import compile_for_backend
from kagome_vqe.vqe import EnergyObjective, prepare, run_vqe

TARGET = np.array([1.5, -0.5, 2.0])


def quadratic(x: np.ndarray) -> float:
    return float(np.sum((x - TARGET) ** 2))


# -- unsharing ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ansatz", "occurrences"),
    [
        (HVAnsatz(), 36),
        (HVAnsatz(reps=2), 72),
        (RVBAnsatz(), 12),
        (LibraryAnsatz("real_amplitudes"), 12),
    ],
)
def test_unshare_counts_and_state(
    ansatz: Ansatz, occurrences: int, hamiltonian: SparsePauliOp, rng: np.random.Generator
) -> None:
    circuit = ansatz.build()
    unshared, mapping = unshare_parameters(circuit)
    assert mapping.num_occurrences == occurrences
    assert mapping.num_parameters == ansatz.num_parameters
    assert unshared.num_parameters == occurrences
    theta = rng.uniform(-1, 1, ansatz.num_parameters)
    shared_problem = compile_for_backend(circuit, hamiltonian, None)
    unshared_problem = compile_for_backend(unshared, hamiltonian, None)
    with StatevectorEnergy(shared_problem) as a, StatevectorEnergy(unshared_problem) as b:
        e_shared = a.evaluate(theta[None, :]).energies[0]
        e_unshared = b.evaluate(mapping.expand(theta)[None, :]).energies[0]
    assert e_unshared == pytest.approx(e_shared, abs=1e-10)


def test_legacy_eswap_cannot_be_unshared() -> None:
    with pytest.raises(ValueError, match="parameter shift"):
        unshare_parameters(RVBAnsatz(gate="legacy").build())


def test_parameter_map_algebra() -> None:
    mapping = ParameterMap(np.array([0, 1, 0, 1, 1]), 2)
    assert not mapping.is_identity
    assert mapping.expand(np.array([3.0, 4.0])).tolist() == [3, 4, 3, 4, 4]
    assert mapping.expand(np.array([[1.0, 2.0], [5.0, 6.0]])).shape == (2, 5)
    assert mapping.contract(np.ones(5)).tolist() == [2.0, 3.0]
    assert ParameterMap.identity(3).is_identity


# -- parameter shift ----------------------------------------------------------------------------


@pytest.mark.parametrize("ansatz", [HVAnsatz(), RVBAnsatz()])
def test_parameter_shift_matches_finite_differences(
    ansatz: Ansatz, hamiltonian: SparsePauliOp, rng: np.random.Generator
) -> None:
    unshared, mapping = unshare_parameters(ansatz.build())
    shared_problem = compile_for_backend(ansatz.build(), hamiltonian, None)
    unshared_problem = compile_for_backend(unshared, hamiltonian, None)
    theta = rng.uniform(-1, 1, ansatz.num_parameters)
    with StatevectorEnergy(unshared_problem) as est_u, StatevectorEnergy(shared_problem) as est_s:
        ps = ParameterShift(mapping).estimate(EnergyObjective(est_u, parameter_map=mapping), theta)
        fd = CentralDifference(1e-5).estimate(EnergyObjective(est_s), theta)
    assert ps.nfev == 2 * mapping.num_occurrences
    assert ps.grad == pytest.approx(fd.grad, abs=1e-6)


def test_parameter_shift_evaluation_counts() -> None:
    assert ParameterShift(unshare_parameters(HVAnsatz().build())[1]).evaluations(2) == 72
    assert ParameterShift(unshare_parameters(RVBAnsatz().build())[1]).evaluations(12) == 24
    assert ParameterShift().evaluations(5) == 10


def test_parameter_shift_needs_expandable_objective_for_shared_parameters() -> None:
    mapping = ParameterMap(np.array([0, 0]), 1)
    with pytest.raises(TypeError, match="ExpandableObjective"):
        ParameterShift(mapping).estimate(as_objective(quadratic), np.zeros(1))


def test_parameter_shift_is_exact_on_sinusoids() -> None:
    def fn(x: np.ndarray) -> float:
        return float(np.sum(np.cos(x)) + np.sin(x[0]) * 0.5)

    x = np.array([0.3, -1.2])
    est = ParameterShift().estimate(as_objective(fn), x)
    expected = -np.sin(x) + np.array([0.5 * np.cos(x[0]), 0.0])
    assert est.grad == pytest.approx(expected, abs=1e-12)


# -- other estimators and update rules ----------------------------------------------------------


def test_spsa_gradient_is_unbiased_on_average() -> None:
    est = SimultaneousPerturbation(0.1, seed=0)
    x = np.zeros(3)
    grads = np.array([est.estimate(as_objective(quadratic), x).grad for _ in range(4000)])
    assert grads.mean(axis=0) == pytest.approx(2 * (x - TARGET), abs=0.25)


def test_finite_difference_accounting() -> None:
    fwd = ForwardDifference(1e-4).estimate(as_objective(quadratic), np.zeros(3))
    assert fwd.nfev == 4
    assert fwd.value == pytest.approx(quadratic(np.zeros(3)))
    assert fwd.grad == pytest.approx(-2 * TARGET, abs=1e-3)


@pytest.mark.parametrize(
    "optimizer",
    [
        Adam(maxiter=600, lr=0.05, gradient=SimultaneousPerturbation(0.05, seed=1)),
        GradientDescent(maxiter=200, lr=0.1, gradient=CentralDifference(1e-4)),
        GradientDescent(maxiter=200, lr=0.1, gradient=ForwardDifference(1e-6)),
    ],
)
def test_new_update_rules_converge(optimizer: Adam | GradientDescent) -> None:
    res = optimizer.minimize(quadratic, np.zeros(3))
    assert res.x == pytest.approx(TARGET, abs=0.1)


def test_optimizer_config_labels_and_builds(rng: np.random.Generator) -> None:
    hybrid = OptimizerConfig(type="adam", gradient="spsa", perturbation=0.1)
    assert hybrid.label() == "adam_spsa"
    assert isinstance(hybrid.build(rng), Adam)
    assert hybrid.build(rng).evaluations_per_iteration(12) == 2
    ps = OptimizerConfig(type="adam", gradient="parameter_shift")
    assert ps.uses_parameter_shift
    assert ps.label() == "adam_parameter_shift"
    gd = OptimizerConfig(type="gd", gradient="parameter_shift")
    assert isinstance(gd.build(rng), GradientDescent)
    assert OptimizerConfig(type="adam").label() == "adam"
    assert isinstance(OptimizerConfig(type="spsa", gradient="parameter_shift").build(rng), Spsa)
    assert not OptimizerConfig(type="spsa", gradient="parameter_shift").uses_parameter_shift


# -- benchmark plumbing -------------------------------------------------------------------------


def test_benchmark_seeds_and_combo_shots() -> None:
    bench = BenchmarkConfig(
        name="m1",
        seeds=[1, 2],
        backend=BackendConfig(kind="aer_ideal", shots=1000),
        combos=[
            ComboConfig(ansatz=AnsatzConfig(type="hva"), optimizer=OptimizerConfig(type="spsa")),
            ComboConfig(
                ansatz=AnsatzConfig(type="hva"),
                optimizer=OptimizerConfig(type="adam", gradient="parameter_shift"),
                shots=50,
            ),
        ],
    )
    runs = bench.run_configs()
    assert [r.name for r in runs] == [
        "m1-hva_spsa-s1",
        "m1-hva_spsa-s2",
        "m1-hva_adam_parameter_shift-s1",
        "m1-hva_adam_parameter_shift-s2",
    ]
    assert [r.backend.shots for r in runs] == [1000, 1000, 50, 50]
    assert [r.seed for r in runs] == [1, 2, 1, 2]


def test_prepare_projects_parameter_shift_cost() -> None:
    cfg = RunConfig(optimizer=OptimizerConfig(type="adam", maxiter=4, gradient="parameter_shift"))
    prepared = prepare(cfg)
    assert prepared.parameter_map.num_occurrences == 36
    assert prepared.projected_evaluations == 4 * 72 + 1
    assert prepared.summary()["ansatz"]["circuit_parameters"] == 36


def test_run_with_parameter_shift_and_ideal_tracking(tmp_path: Path) -> None:
    cfg = RunConfig(
        name="ps",
        seed=4,
        optimizer=OptimizerConfig(type="adam", maxiter=3, lr=0.05, gradient="parameter_shift"),
        backend=BackendConfig(kind="aer_ideal", shots=64, final_shots=256),
        output_dir=tmp_path,
    )
    result = run_vqe(cfg)
    assert result.nfev == 3 * 72 + 1
    assert len(result.ideal_trace) == 4  # initial point plus one per iteration
    shots = [p.shots for p in result.ideal_trace]
    assert shots == sorted(shots)
    assert shots[1] == 3 * 72 * 64  # three bases, one gradient
    assert result.ideal_energy is not None
    assert result.ideal_energy >= result.exact_energy - 1e-9
    assert len(result.params_history) == result.nfev
    assert all(p.shape == (2,) for p in result.params_history)
    loaded = load_run(save_run(result, new_run_dir(cfg)))
    assert loaded.ideal_trace is not None
    assert len(loaded.ideal_trace) == 4
    assert loaded.result["ideal_energy"] == pytest.approx(result.ideal_energy)


def test_ideal_tracking_can_be_disabled() -> None:
    cfg = RunConfig(optimizer=OptimizerConfig(type="spsa", maxiter=2), track_ideal_energy=False)
    result = run_vqe(cfg)
    assert result.ideal_trace == []
    assert result.ideal_energy is None


def test_aggregate_summary_and_ideal_plot(tmp_path: Path) -> None:
    bench = BenchmarkConfig(
        name="agg",
        seeds=[1, 2],
        backend=BackendConfig(kind="aer_ideal", shots=64, final_shots=64),
        output_dir=tmp_path,
        combos=[
            ComboConfig(
                ansatz=AnsatzConfig(type="hva"), optimizer=OptimizerConfig(type="spsa", maxiter=2)
            )
        ],
    )
    runs = [load_run(save_run(run_vqe(cfg), new_run_dir(cfg))) for cfg in bench.run_configs()]
    assert {r.combo for r in runs} == {"agg-hva_spsa"}
    _, md_path = write_benchmark_summary(runs, tmp_path / "summary")
    table = md_path.read_text()
    assert "Aggregated over seeds" in table
    assert "| agg-hva_spsa | 2 |" in table
    fig = plot_ideal_vs_shots(runs, out=tmp_path / "ideal.png")
    assert fig is not None
    assert (tmp_path / "ideal.png").exists()
    cli = CliRunner().invoke(
        app,
        ["plot", *(str(r.path) for r in runs), "--kind", "ideal", "--out", str(tmp_path / "c.png")],
    )
    assert cli.exit_code == 0, cli.output
    assert (tmp_path / "c.png").exists()
