"""Persist runs to disk and load them back.

A results directory ``results/<UTC timestamp>-<name>/`` contains::

    config.toml        the resolved RunConfig (replay with `kagome-vqe run config.toml`)
    result.json        final energies, errors, parameters, ISA stats, timings
    trace.csv          one row per energy evaluation
    params.npy         parameters of every evaluation (optional, shape (nfev, n))
    ideal_trace.csv    noise-free energy after every iteration vs shots spent (optional)
    environment.json   package versions, Python, platform, git commit
    run.log            the log of the run (attached by the CLI)

Everything is plain text or numpy so it can be analysed without this package.
"""

from __future__ import annotations

import json
import logging
import platform
import re
import subprocess
import sys
from collections.abc import Hashable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tomli_w

from kagome_vqe.config import RunConfig, load_toml
from kagome_vqe.vqe import TraceRow, VQERunResult

__all__ = [
    "LoadedRun",
    "environment_info",
    "load_run",
    "new_run_dir",
    "save_run",
    "write_benchmark_summary",
    "write_evaluation_summary",
]

log = logging.getLogger(__name__)

_TRACKED_PACKAGES = (
    "kagome-vqe",
    "qiskit",
    "qiskit-aer",
    "qiskit-ibm-runtime",
    "rustworkx",
    "numpy",
    "scipy",
)


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def environment_info() -> dict[str, Any]:
    """Versions and provenance to store next to every result."""
    versions: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    sha = _git("rev-parse", "HEAD")
    dirty = None if sha is None else bool(_git("status", "--porcelain"))
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "git": {"commit": sha, "dirty": dirty},
        "argv": sys.argv,
    }


def new_run_dir(cfg: RunConfig, root: Path | None = None) -> Path:
    """Create ``<root>/<UTC timestamp>-<name>/`` and return it."""
    root = Path(root or cfg.output_dir)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = root / f"{stamp}-{cfg.name}"
    counter = 1
    while path.exists():
        counter += 1
        path = root / f"{stamp}-{cfg.name}-{counter}"
    path.mkdir(parents=True)
    return path


def _result_dict(result: VQERunResult) -> dict[str, Any]:
    return {
        "name": result.config.name,
        "final_energy": result.final_energy,
        "final_std": result.final_std,
        "final_shots": result.final_shots,
        "final_relative_error": result.final_relative_error,
        "within_1_percent": result.final_relative_error < 0.01,
        "exact_energy": result.exact_energy,
        "energy": result.energy,
        "relative_error": result.relative_error,
        "best_energy": result.best_energy,
        "best_relative_error": result.best_relative_error,
        "total_shots": result.total_shots,
        "ideal_energy": result.ideal_energy,
        "ideal_relative_error": result.ideal_relative_error,
        "nfev": result.nfev,
        "nit": result.nit,
        "wall_time_s": result.wall_time_s,
        "optimizer_message": result.optimizer_message,
        "x0": result.x0.tolist(),
        "optimal_params": result.optimal_params.tolist(),
        "ansatz": result.config.ansatz.type,
        "optimizer": result.config.optimizer.type,
        "backend": result.config.backend.label(),
        "isa": result.isa_stats,
        "estimator": result.estimator,
    }


def save_run(result: VQERunResult, run_dir: Path) -> Path:
    """Write all artefacts of ``result`` into ``run_dir`` (created if needed)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.toml").open("wb") as fh:
        tomli_w.dump(result.config.to_toml_dict(), fh)
    (run_dir / "result.json").write_text(json.dumps(_result_dict(result), indent=2) + "\n")
    pd.DataFrame([asdict(r) for r in result.trace]).to_csv(run_dir / "trace.csv", index=False)
    if result.params_history:
        np.save(run_dir / "params.npy", np.vstack(result.params_history))
    if result.ideal_trace:
        pd.DataFrame([asdict(p) for p in result.ideal_trace]).to_csv(
            run_dir / "ideal_trace.csv", index=False
        )
    (run_dir / "environment.json").write_text(json.dumps(environment_info(), indent=2) + "\n")
    log.info("saved run to %s", run_dir)
    return run_dir


@dataclass(slots=True)
class LoadedRun:
    """A run read back from disk."""

    path: Path
    config: RunConfig
    result: dict[str, Any]
    trace: pd.DataFrame
    params: np.ndarray | None
    ideal_trace: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        """Run name."""
        return str(self.result["name"])

    @property
    def combo(self) -> str:
        """Run name without a trailing ``-s<seed>``: the label shared by repeated seeds."""
        return re.sub(r"-s\d+$", "", self.name)

    def energies(self) -> np.ndarray:
        """Energy per evaluation."""
        return self.trace["energy"].to_numpy(dtype=float)

    def energies_per_iteration(self) -> np.ndarray:
        """Lowest energy per optimiser iteration."""
        grouped = self.trace.groupby("iteration")["energy"].min().sort_index()
        return grouped.to_numpy(dtype=float)

    def rows(self) -> list[TraceRow]:
        """Trace as :class:`~kagome_vqe.vqe.TraceRow` objects."""
        records: list[dict[Hashable, Any]] = self.trace.to_dict("records")
        return [
            TraceRow(
                eval_idx=int(r["eval_idx"]),
                iteration=int(r["iteration"]),
                energy=float(r["energy"]),
                std=float(r["std"]),
                shots=None if pd.isna(r["shots"]) else int(r["shots"]),
                wall_time_s=float(r["wall_time_s"]),
                job_id=None if pd.isna(r["job_id"]) else str(r["job_id"]),
            )
            for r in records
        ]


def load_run(run_dir: Path) -> LoadedRun:
    """Read a results directory."""
    run_dir = Path(run_dir)
    config = RunConfig.model_validate(load_toml(run_dir / "config.toml"))
    result = json.loads((run_dir / "result.json").read_text())
    trace = pd.read_csv(run_dir / "trace.csv")
    params_path = run_dir / "params.npy"
    params = np.load(params_path) if params_path.exists() else None
    ideal_path = run_dir / "ideal_trace.csv"
    ideal = pd.read_csv(ideal_path) if ideal_path.exists() else None
    return LoadedRun(
        path=run_dir, config=config, result=result, trace=trace, params=params, ideal_trace=ideal
    )


def _summary_row(run: LoadedRun) -> dict[str, Any]:
    """One benchmark row; tolerates result files written before the re-measurement existed."""
    res = run.result
    final_energy = res.get("final_energy", res["energy"])
    return {
        "name": run.name,
        "ansatz": res["ansatz"],
        "optimizer": res["optimizer"],
        "backend": res["backend"],
        "final_energy": final_energy,
        "final_std": res.get("final_std", 0.0),
        "final_shots": res.get("final_shots"),
        "exact_energy": res["exact_energy"],
        "final_relative_error": res.get("final_relative_error", res["relative_error"]),
        "best_energy": res["best_energy"],
        "two_qubit_gates": res.get("isa", {}).get("two_qubit_gates"),
        "nfev": res["nfev"],
        "total_shots": res.get("total_shots", 0),
        "ideal_energy": res.get("ideal_energy"),
        "ideal_relative_error": res.get("ideal_relative_error"),
        "wall_time_s": res["wall_time_s"],
        "path": str(run.path),
    }


def write_benchmark_summary(runs: list[LoadedRun], out_dir: Path) -> tuple[Path, Path]:
    """Write ``benchmark.json`` and ``benchmark.md`` (a Markdown table) for a set of runs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [_summary_row(run) for run in runs]
    json_path = out_dir / "benchmark.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    header = (
        "| run | ansatz | optimizer | backend | final E (re-measured) | reported std | exact E "
        "| rel. err | 2q gates | evaluations | total shots | time (s) |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = "".join(
        f"| {r['name']} | {r['ansatz']} | {r['optimizer']} | {r['backend']} | "
        f"{r['final_energy']:.4f} | {r['final_std']:.3f} | {r['exact_energy']:.4f} | "
        f"{100 * r['final_relative_error']:.3f} % | {r['two_qubit_gates']} | {r['nfev']} | "
        f"{r['total_shots']:,} | {r['wall_time_s']:.1f} |\n"
        for r in rows
    )
    md_path = out_dir / "benchmark.md"
    md_path.write_text(header + body + _aggregate_table(runs))
    return json_path, md_path


def _sample_std(values: np.ndarray) -> float:
    """Sample standard deviation (``ddof=1``) ignoring NaN; 0 for a single value."""
    finite = values[np.isfinite(values)]
    return float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0


def _aggregate_table(runs: list[LoadedRun]) -> str:
    """Per-combo statistics over seeds, when any combo was repeated; empty string otherwise."""
    groups: dict[str, list[LoadedRun]] = {}
    for run in runs:
        groups.setdefault(run.combo, []).append(run)
    if all(len(g) == 1 for g in groups.values()):
        return ""
    lines = [
        "",
        "Aggregated over seeds (mean and sample standard deviation). The noise-free column is the",
        "statevector energy of the final parameters: parameter quality, independent of noise.",
        "",
        "| combo | seeds | final E (re-measured) | noise-free E of final params "
        "| noise-free rel. err | evaluations | total shots |",
        "|---|---|---|---|---|---|---|",
    ]
    for combo, group in groups.items():
        rows = [_summary_row(r) for r in group]
        final = np.array([r["final_energy"] for r in rows], dtype=float)
        ideal = np.array([np.nan if r["ideal_energy"] is None else r["ideal_energy"] for r in rows])
        ideal_err = np.array(
            [
                np.nan if r["ideal_relative_error"] is None else r["ideal_relative_error"]
                for r in rows
            ]
        )
        nfev = np.mean([r["nfev"] for r in rows])
        shots = np.mean([r["total_shots"] for r in rows])
        lines.append(
            f"| {combo} | {len(rows)} | {final.mean():.4f} +- {_sample_std(final):.4f} | "
            f"{np.nanmean(ideal):.4f} +- {_sample_std(ideal):.4f} | "
            f"{100 * np.nanmean(ideal_err):.3f} % | {nfev:.0f} | {shots:,.0f} |"
        )
    return "\n".join(lines) + "\n"


def write_evaluation_summary(summaries: list[dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    """Write ``evaluation.json`` and ``evaluation.md`` for fixed-point (mitigation) studies.

    Each summary is :meth:`kagome_vqe.vqe.FixedPointResult.summary`. The table shows the bias
    of the mean against the noise-free energy and the real spread between repeats; values that
    lie significantly below the exact ground energy are flagged, since by the variational
    principle they are artefacts.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "evaluation.json"
    json_path.write_text(json.dumps(summaries, indent=2, default=str) + "\n")
    header = (
        "| run | ansatz | mitigation | noise-free E | mean E | spread | bias | rel. bias "
        "| shots per energy | setup shots | below ground? |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = "".join(
        f"| {r['name']} | {r['ansatz']} | {r['mitigation']} | {r['ideal_energy']:.4f} | "
        f"{r['mean']:.4f} | {r['spread']:.4f} | {r['bias']:+.4f} | "
        f"{100 * r['relative_bias']:.2f} % | {r['shots_per_evaluation']:,} | "
        f"{r['setup_shots']:,} | {'YES' if r['below_ground'] else 'no'} |\n"
        for r in summaries
    )
    md_path = out_dir / "evaluation.md"
    md_path.write_text(header + body)
    return json_path, md_path
