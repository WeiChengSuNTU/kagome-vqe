"""Run persistence (run_and_save, status, reload, replay, failures) and multi-seed studies."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from kagome_vqe.cli import app
from kagome_vqe.config import (
    AnsatzConfig,
    BackendConfig,
    BenchmarkConfig,
    BudgetConfig,
    ComboConfig,
    LatticeConfig,
    OptimizerConfig,
    RunConfig,
    load_run_config,
)
from kagome_vqe.results import (
    RESULT_SCHEMA_VERSION,
    RunFailedError,
    find_runs,
    load_optimal_params,
    load_run,
    read_status,
    run_and_save,
    write_benchmark_summary,
)
from kagome_vqe.study import (
    StudyBackend,
    StudyConfig,
    estimate_study,
    fitted_maxiter,
    load_study_config,
    run_study,
)
from kagome_vqe.vqe import TraceRow

runner = CliRunner()
ROOT = Path(__file__).resolve().parents[2]
STAR = LatticeConfig(kind="kagome_unit_cell")


def _status(run_dir: Path) -> dict[str, Any]:
    status = read_status(run_dir)
    assert status is not None
    return status


def _graph_run(tmp_path: Path, **update: Any) -> RunConfig:
    cfg = RunConfig(
        name="persist",
        seed=3,
        x0_scale=0.1,
        lattice=STAR,
        ansatz=AnsatzConfig(type="graph_hva", reps=1),
        optimizer=OptimizerConfig(type="spsa", calibrate=True, calibration_samples=3, maxiter=50),
        backend=BackendConfig(kind="statevector"),
        budget=BudgetConfig(max_evaluations=21),
        output_dir=tmp_path / "results",
    )
    return cfg.model_copy(update=update)


# -- run_and_save -----------------------------------------------------------------------------


def test_run_and_save_writes_complete_reloadable_artifacts(tmp_path: Path) -> None:
    cfg = _graph_run(tmp_path)
    run_dir = run_and_save(cfg)
    for name in (
        "config.toml",
        "result.json",
        "trace.csv",
        "params.npy",
        "ideal_trace.csv",
        "environment.json",
        "run.log",
        "status.json",
    ):
        assert (run_dir / name).exists(), name
    assert _status(run_dir)["status"] == "completed"
    run = load_run(run_dir)
    res = run.result
    assert res["schema_version"] == RESULT_SCHEMA_VERSION
    assert run.status == "completed"
    # parameters: dimensions agree everywhere
    n = res["num_parameters"]
    assert n == 18  # 18 bonds of the star, one parameter per gate, no H0
    assert run.optimal_params.shape == run.x0.shape == (n,)
    assert run.params is not None
    assert run.params.shape == (res["nfev"], n)
    np.testing.assert_array_equal(
        load_optimal_params(run_dir, num_parameters=n), run.optimal_params
    )
    with pytest.raises(ValueError, match="parameters"):
        load_optimal_params(run_dir, num_parameters=n + 1)
    # resources: budget respected, calibration counted, trace consistent
    resources = run.resources
    assert res["nfev"] == len(run.trace) <= 21
    assert resources["calibration_evaluations"] == 6
    assert resources["optimizer_evaluations"] == res["nfev"] - 6
    assert resources["budget"]["max_evaluations"] == 21
    assert resources["total_shots"] == 0
    assert resources["optimiser_wall_time_s"] <= resources["wall_time_s"]
    # metadata: the built lattice and ansatz in full
    meta = run.metadata
    assert meta["lattice"]["num_bonds"] == len(meta["lattice"]["bonds"]) == 18
    assert meta["lattice"]["boundary"] == "open"
    assert len(meta["lattice"]["dimers"]) == 6
    assert meta["ansatz"]["include_h0"] is False
    assert meta["ansatz"]["num_parameters"] == n
    assert "calibrated gains" in meta["optimizer"]["gradient"]
    assert meta["termination"]["reason"] in ("optimizer", "budget")
    # the config round-trips and replays to the identical result
    assert load_run_config(run_dir / "config.toml") == cfg
    replay = load_run(run_and_save(load_run_config(run_dir / "config.toml")))
    assert replay.path != run.path
    assert replay.result["final_energy"] == res["final_energy"]
    np.testing.assert_array_equal(replay.optimal_params, run.optimal_params)


def test_budget_stop_keeps_last_iterate(tmp_path: Path) -> None:
    cfg = _graph_run(
        tmp_path,
        optimizer=OptimizerConfig(type="scipy", method="L-BFGS-B", maxiter=1000),
        budget=BudgetConfig(max_evaluations=45),
    )
    run = load_run(run_and_save(cfg))
    assert run.result["nfev"] <= 45
    assert run.resources["budget_exhausted"] is True
    assert run.metadata["termination"]["reason"] == "budget"
    assert "budget exhausted" in run.result["optimizer_message"]
    ideal = run.ideal_trace
    assert ideal is not None
    last = ideal["ideal_energy"].iloc[-1]
    assert run.result["final_energy"] == pytest.approx(last)  # the last completed iterate


def test_shot_accounting_on_a_sampling_backend(tmp_path: Path) -> None:
    """aer_ideal: 3 measurement bases; the shot budget binds; the re-measurement is added."""
    cfg = _graph_run(
        tmp_path,
        optimizer=OptimizerConfig(type="adam", gradient="spsa", maxiter=100, perturbation=0.1),
        backend=BackendConfig(kind="aer_ideal", shots=64, final_shots=256),
        budget=BudgetConfig(max_shots=64 * 3 * 9),  # nine evaluations
    )
    run = load_run(run_and_save(cfg))
    res = run.resources
    assert res["measurement_groups"] == 3
    assert run.result["nfev"] == 8  # ADAM+SPSA requests pairs: 4 pairs fit, a 5th would not
    assert res["optimisation_shots"] == 8 * 64 * 3 <= 64 * 3 * 9
    assert res["total_shots"] == run.result["total_shots"] == 8 * 64 * 3 + 256 * 3
    assert res["budget_exhausted"] is True
    assert run.result["ideal_energy"] is not None  # parameter quality, free of shot noise
    assert run.trace["shots"].eq(64).all()


def test_failed_run_keeps_partial_metadata_and_never_overwrites(tmp_path: Path) -> None:
    cfg = _graph_run(tmp_path)
    seen: list[TraceRow] = []

    def explode(row: TraceRow) -> None:
        seen.append(row)
        if len(seen) == 4:
            msg = "simulated backend failure"
            raise RuntimeError(msg)

    with pytest.raises(RunFailedError) as info:
        run_and_save(cfg, on_evaluation=explode)
    failed = info.value.run_dir
    status = _status(failed)
    assert status["status"] == "failed"
    assert "simulated backend failure" in status["error"]
    assert status["completed_evaluations"] == 4
    assert (failed / "config.toml").exists()
    assert (failed / "environment.json").exists()
    assert len(pd.read_csv(failed / "partial_trace.csv")) == 4
    assert not (failed / "result.json").exists()
    with pytest.raises(FileNotFoundError, match="failed"):
        load_run(failed)
    # a second attempt gets a new directory; the failed one is untouched
    ok = run_and_save(cfg)
    assert ok != failed
    assert _status(failed)["status"] == "failed"
    assert [p.name for p in find_runs(tmp_path)] == [ok.name]
    assert {p.name for p in find_runs(tmp_path, status=None)} == {ok.name, failed.name}


def test_schema1_directories_still_load(tmp_path: Path) -> None:
    """Directories written before status.json / resources / metadata existed."""
    new = run_and_save(_graph_run(tmp_path))
    old = tmp_path / "old-run"
    shutil.copytree(new, old)
    (old / "status.json").unlink()
    data = json.loads((old / "result.json").read_text())
    for key in ("schema_version", "status", "seed", "num_parameters", "resources", "metadata"):
        data.pop(key)
    (old / "result.json").write_text(json.dumps(data))
    run = load_run(old)
    assert run.status == "completed"
    assert run.resources == {}
    assert run.metadata == {}
    assert run.optimal_params.size == 18
    json_path, _ = write_benchmark_summary([run, load_run(new)], tmp_path / "summary")
    assert len(json.loads(json_path.read_text())) == 2
    assert old in list(find_runs(tmp_path))


def test_legacy_hva_run_unchanged_by_new_fields(tmp_path: Path) -> None:
    cfg = RunConfig(
        name="legacy",
        seed=1,
        x0_scale=0.2,
        optimizer=OptimizerConfig(type="spsa", maxiter=3),
        backend=BackendConfig(kind="statevector"),
        output_dir=tmp_path,
    )
    run = load_run(run_and_save(cfg))
    assert run.result["ansatz"] == "hva"
    assert run.result["nfev"] == 2 * 3 + 1
    assert run.resources["budget_exhausted"] is False
    assert run.metadata["ansatz"]["num_parameters"] == 2


def test_benchmark_config_carries_budget() -> None:
    bench = BenchmarkConfig(
        seeds=[1, 2],
        budget=BudgetConfig(max_evaluations=9),
        combos=[ComboConfig(ansatz=AnsatzConfig(type="hva"))],
    )
    runs = bench.run_configs()
    assert [r.name for r in runs] == ["benchmark-hva_adam-s1", "benchmark-hva_adam-s2"]
    assert all(r.budget.max_evaluations == 9 for r in runs)


# -- studies ----------------------------------------------------------------------------------


def _tiny_study(tmp_path: Path) -> StudyConfig:
    return StudyConfig.model_validate(
        {
            "name": "tiny",
            "output_dir": str(tmp_path / "results"),
            "seeds": [1, 2],
            "x0_scale": 0.1,
            "reps": [1],
            "lattices": [{"label": "star", "lattice": {"kind": "kagome_unit_cell"}}],
            "backends": [
                {
                    "label": "sv",
                    "backend": {"kind": "statevector"},
                    "budget": {"max_evaluations": 30},
                },
            ],
            "optimizers": [
                {
                    "label": "lbfgsb",
                    "optimizer": {"type": "scipy", "method": "L-BFGS-B", "maxiter": 1000},
                },
                {
                    "label": "spsa_cal",
                    "fit_maxiter_to_budget": True,
                    "optimizer": {
                        "type": "spsa",
                        "calibrate": True,
                        "calibration_samples": 3,
                        "perturbation": 0.1,
                        "target_magnitude": 0.2,
                    },
                },
                {
                    "label": "adam_spsa",
                    "fit_maxiter_to_budget": True,
                    "optimizer": {
                        "type": "adam",
                        "gradient": "spsa",
                        "lr": 0.02,
                        "perturbation": 0.1,
                    },
                },
            ],
        }
    )


def test_fitted_maxiter_spends_the_budget_exactly() -> None:
    sv = StudyBackend(label="sv", backend=BackendConfig(), budget=BudgetConfig(max_evaluations=301))
    spsa = OptimizerConfig(type="spsa", calibrate=True, calibration_samples=25)
    assert fitted_maxiter(spsa, sv, STAR, 18) == (301 - 50 - 1) // 2
    assert fitted_maxiter(OptimizerConfig(type="spsa"), sv, STAR, 18) == 150
    assert fitted_maxiter(OptimizerConfig(type="adam", gradient="spsa"), sv, STAR, 18) == 150
    assert fitted_maxiter(OptimizerConfig(type="adam"), sv, STAR, 18) == 300 // 19
    assert fitted_maxiter(OptimizerConfig(type="scipy"), sv, STAR, 18) is None
    shots = StudyBackend(
        label="s",
        backend=BackendConfig(kind="aer_ideal", shots=100),
        budget=BudgetConfig(max_shots=3 * 100 * 41),  # 41 evaluations at 3 bases
    )
    assert fitted_maxiter(OptimizerConfig(type="adam", gradient="spsa"), shots, STAR, 18) == 20
    tiny = StudyBackend(label="t", backend=BackendConfig(), budget=BudgetConfig(max_evaluations=10))
    with pytest.raises(ValueError, match="no iteration"):
        fitted_maxiter(spsa, tiny, STAR, 18)


def test_study_expansion_names_and_fair_starts(tmp_path: Path) -> None:
    cfg = _tiny_study(tmp_path)
    cells = cfg.runs()
    names = [c.config.name for c in cells]
    assert len(names) == len(set(names)) == 6
    assert all(n.endswith(("-s1", "-s2")) for n in names)
    assert names[0] == "tiny-star-p1-noh0-sv-lbfgsb-s1"
    # ordered so that each seed's optimiser comparison is contiguous
    assert [c.optimizer for c in cells[:3]] == ["lbfgsb", "spsa_cal", "adam_spsa"]
    assert {c.config.seed for c in cells[:3]} == {1}
    spsa = next(c for c in cells if c.optimizer == "spsa_cal")
    assert spsa.config.optimizer.maxiter == (30 - 6 - 1) // 2
    with pytest.raises(ValueError, match="duplicate"):
        StudyConfig.model_validate({**cfg.model_dump(), "optimizers": [cfg.optimizers[0]] * 2})


def test_run_study_persists_summarises_and_resumes(tmp_path: Path) -> None:
    cfg = _tiny_study(tmp_path)
    study_dir = run_study(cfg, max_runs=4)
    manifest = [json.loads(x) for x in (study_dir / "manifest.jsonl").read_text().splitlines()]
    assert [m["status"] for m in manifest] == ["completed"] * 4
    assert load_study_config(study_dir / "study.toml") == cfg
    run_study(cfg, study_dir=study_dir)  # resume: only the two missing runs
    manifest = [json.loads(x) for x in (study_dir / "manifest.jsonl").read_text().splitlines()]
    assert len(manifest) == 6
    assert len({m["name"] for m in manifest}) == 6
    runs = [load_run(study_dir / m["path"]) for m in manifest]
    # identical initial parameters for every optimiser of a seed
    for seed in (1, 2):
        starts = [r.x0 for r in runs if r.config.seed == seed]
        assert len(starts) == 3
        for x0 in starts[1:]:
            np.testing.assert_array_equal(x0, starts[0])
    assert all(r.result["nfev"] <= 30 for r in runs)
    summary = study_dir / "summary"
    table = pd.read_csv(summary / "runs.csv")
    groups = pd.read_csv(summary / "groups.csv")
    assert len(table) == 6
    assert len(groups) == 3
    assert set(groups["seeds"]) == {2}
    assert (groups["success_rate"].between(0, 1)).all()
    report = (summary / "report.md").read_text()
    assert "Aggregated over seeds" in report
    assert "simultaneous perturbation" in report
    assert sorted(p.name for p in (summary / "figures").iterdir()) == [
        "error_vs_evaluations-sv.png",
        "fidelity_vs_depth-star.png",
        "seeds-star.png",
    ]
    # resuming with a different configuration is refused
    with pytest.raises(ValueError, match="differs"):
        run_study(cfg.model_copy(update={"seeds": [1, 2, 3]}), study_dir=study_dir)


def test_estimate_study_reports_budget_and_time(tmp_path: Path) -> None:
    frame, totals = estimate_study(_tiny_study(tmp_path), probe_evaluations=2)
    assert totals["runs"] == 6
    assert totals["evaluations"] == 6 * 30
    assert (frame["seconds_per_evaluation"] > 0).all()


def test_shipped_study_profiles_validate() -> None:
    sizes = {}
    for path in sorted((ROOT / "configs" / "studies").glob("*.toml")):
        sizes[path.stem] = len(load_study_config(path).runs())
    assert sizes["graph_hva_optimizer_smoke"] == 5
    assert sizes["graph_hva_baseline"] == 36
    assert sizes["graph_hva_optimizer_full"] >= 5 * 3 * 3  # >= 5 seeds per comparison group


# -- command line -----------------------------------------------------------------------------


def test_cli_params_and_replay(tmp_path: Path) -> None:
    run_dir = run_and_save(_graph_run(tmp_path))
    shown = runner.invoke(app, ["params", str(run_dir)])
    assert shown.exit_code == 0, shown.output
    assert '"num_parameters": 18' in shown.output
    out = tmp_path / "theta.npy"
    saved = runner.invoke(app, ["params", str(run_dir), "--out", str(out)])
    assert saved.exit_code == 0, saved.output
    np.testing.assert_array_equal(np.load(out), load_optimal_params(run_dir))
    replayed = runner.invoke(app, ["replay", str(run_dir), "--no-progress"])
    assert replayed.exit_code == 0, replayed.output
    assert "difference +0.00e+00" in replayed.output
    assert len(list(find_runs(tmp_path))) == 2


def test_cli_study_dry_run_and_output_dir(tmp_path: Path) -> None:
    cfg_path = tmp_path / "tiny.toml"
    import tomli_w

    cfg_path.write_bytes(tomli_w.dumps(_tiny_study(tmp_path).to_toml_dict()).encode())
    plan = runner.invoke(app, ["study", str(cfg_path), "--dry-run"])
    assert plan.exit_code == 0, plan.output
    assert '"runs": 6' in plan.output
    assert not (tmp_path / "results").exists()  # a dry run creates nothing
    out = tmp_path / "elsewhere"
    done = runner.invoke(app, ["study", str(cfg_path), "--output-dir", str(out), "--max-runs", "1"])
    assert done.exit_code == 0, done.output
    assert len(list(find_runs(out))) == 1


def test_cli_study_summary_regenerates(tmp_path: Path) -> None:
    study_dir = run_study(_tiny_study(tmp_path), max_runs=3, summarise=False)
    assert not (study_dir / "summary").exists()
    result = runner.invoke(app, ["study-summary", str(study_dir)])
    assert result.exit_code == 0, result.output
    assert (study_dir / "summary" / "report.md").exists()
