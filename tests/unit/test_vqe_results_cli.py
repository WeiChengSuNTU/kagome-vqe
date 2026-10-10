from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from typer.testing import CliRunner

from kagome_vqe.cli import app
from kagome_vqe.config import BackendConfig, OptimizerConfig, RunConfig
from kagome_vqe.estimators import EnergyBatch
from kagome_vqe.plotting import plot_runs
from kagome_vqe.results import (
    environment_info,
    load_run,
    new_run_dir,
    save_run,
    write_benchmark_summary,
)
from kagome_vqe.vqe import EnergyObjective, TraceRow, prepare, run_vqe

runner = CliRunner()
CONFIGS = Path(__file__).resolve().parents[2] / "configs"


@pytest.fixture
def small_cfg(tmp_path: Path) -> RunConfig:
    return RunConfig(
        name="unit",
        seed=1,
        x0_scale=0.2,
        optimizer=OptimizerConfig(type="spsa", maxiter=3),
        backend=BackendConfig(kind="statevector"),
        output_dir=tmp_path / "results",
    )


def test_prepare_reports_projection(small_cfg: RunConfig) -> None:
    prepared = prepare(small_cfg)
    assert prepared.exact_energy == pytest.approx(-18.0)
    assert prepared.projected_evaluations == 2 * 3 + 1
    summary = prepared.summary()
    assert summary["ansatz"] == {"name": "hva", "num_parameters": 2, "circuit_parameters": 2}
    assert summary["lattice"]["ring_bond_fraction"] == 1.0


def test_run_vqe_trace_accounting(small_cfg: RunConfig) -> None:
    seen: list[TraceRow] = []
    result = run_vqe(small_cfg, on_evaluation=seen.append)
    assert result.nfev == 7
    assert len(result.trace) == 7
    assert len(seen) == 7
    assert [r.iteration for r in result.trace] == [1, 1, 2, 2, 3, 3, 4]
    assert [r.eval_idx for r in result.trace] == list(range(7))
    assert result.energies_per_iteration().shape == (4,)
    assert result.best_energy <= result.energy
    assert result.relative_error < 0.2
    assert len(result.params_history) == 7


def test_final_remeasurement_exact(small_cfg: RunConfig) -> None:
    result = run_vqe(small_cfg)
    assert result.final_shots is None
    assert result.total_shots == 0
    assert result.final_std == 0.0
    assert result.final_energy == pytest.approx(result.energy)  # exact: same point, same value
    assert result.final_relative_error == pytest.approx(result.relative_error)


def test_final_remeasurement_and_shot_accounting(tmp_path: Path) -> None:
    cfg = RunConfig(
        name="shots",
        seed=2,
        x0_scale=0.2,
        optimizer=OptimizerConfig(type="spsa", maxiter=2),
        backend=BackendConfig(kind="aer_ideal", shots=200, final_shots=5000),
        output_dir=tmp_path,
    )
    result = run_vqe(cfg)
    assert result.final_shots == 5000
    assert all(row.shots == 200 for row in result.trace)
    assert result.nfev == 5  # the re-measurement is not part of the optimiser's trace
    assert result.total_shots == 3 * (5 * 200 + 5000)  # 3 measurement bases
    assert result.final_std > 0
    assert abs(result.final_energy - result.exact_energy) < 3.0
    run_dir = save_run(result, new_run_dir(cfg))
    loaded = load_run(run_dir)
    assert loaded.result["final_shots"] == 5000
    assert loaded.result["total_shots"] == result.total_shots
    assert loaded.result["within_1_percent"] == (result.final_relative_error < 0.01)


def test_energy_objective_records_batches() -> None:
    class Fake:
        description: ClassVar[dict[str, Any]] = {}

        def evaluate(self, params: np.ndarray, *, shots: int | None = None) -> EnergyBatch:
            del shots
            return EnergyBatch(energies=params.sum(axis=1), stds=np.zeros(len(params)), shots=5)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

    obj = EnergyObjective(Fake())
    assert obj.batch(np.ones((3, 2))).tolist() == [2.0, 2.0, 2.0]
    assert obj(np.zeros(2)) == 0.0
    assert obj.nfev == 4
    assert obj.best is not None
    assert obj.best.energy == 0.0
    assert obj.trace[0].shots == 5


def test_save_and_load_round_trip(small_cfg: RunConfig) -> None:
    result = run_vqe(small_cfg)
    run_dir = new_run_dir(small_cfg)
    assert run_dir.name.endswith("-unit")
    save_run(result, run_dir)
    for name in ("config.toml", "result.json", "trace.csv", "params.npy", "environment.json"):
        assert (run_dir / name).exists()
    loaded = load_run(run_dir)
    assert loaded.config == small_cfg
    assert loaded.result["nfev"] == 7
    assert loaded.energies().shape == (7,)
    assert loaded.params is not None
    assert loaded.params.shape == (7, 2)
    assert [r.iteration for r in loaded.rows()] == [r.iteration for r in result.trace]

    # a second dir with the same name gets a suffix instead of clobbering
    other = new_run_dir(small_cfg)
    assert other != run_dir

    json_path, md_path = write_benchmark_summary([loaded], run_dir.parent / "summary")
    assert json_path.exists()
    table = md_path.read_text()
    assert "rel. err" in table
    assert "total shots" in table
    assert "final E (re-measured)" in table
    fig = plot_runs([loaded], axis="evaluation")
    assert fig.axes[0].lines


def test_environment_info_has_versions() -> None:
    info = environment_info()
    assert info["packages"]["qiskit"]
    assert "git" in info


def test_cli_exact() -> None:
    result = runner.invoke(app, ["exact"])
    assert result.exit_code == 0, result.output
    assert "-18.0000" in result.output
    assert "degeneracy" in result.output


def test_cli_dry_run_with_overrides() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            str(CONFIGS / "statevector_hva_adam.toml"),
            "--dry-run",
            "--set",
            "optimizer.maxiter=3",
            "--backend-override",
            "aer_ideal",
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"projected_evaluations": 10' in result.output
    assert '"backend": "aer_ideal"' in result.output


def test_cli_run_and_plot(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            str(CONFIGS / "statevector_hva_adam.toml"),
            "--set",
            "optimizer.maxiter=2",
            "--set",
            f'output_dir="{tmp_path}"',
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.output
    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "run.log").read_text()
    out = tmp_path / "fig.png"
    result = runner.invoke(app, ["plot", str(run_dirs[0]), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_cli_rejects_bad_override() -> None:
    result = runner.invoke(
        app, ["run", str(CONFIGS / "statevector_hva_adam.toml"), "--set", "nonsense"]
    )
    assert result.exit_code != 0
