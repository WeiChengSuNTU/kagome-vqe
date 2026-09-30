"""Persist runs to disk and load them back.

A results directory ``results/<UTC timestamp>-<name>/`` contains::

    config.toml        the resolved RunConfig (replay with `kagome-vqe run config.toml`)
    result.json        final energies, errors, parameters, ISA stats, timings
    trace.csv          one row per energy evaluation
    params.npy         parameters of every evaluation (optional, shape (nfev, n))
    environment.json   package versions, Python, platform, git commit
    run.log            the log of the run (attached by the CLI)

Everything is plain text or numpy so it can be analysed without this package.
"""

from __future__ import annotations

import json
import logging
import platform
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
        "energy": result.energy,
        "best_energy": result.best_energy,
        "exact_energy": result.exact_energy,
        "relative_error": result.relative_error,
        "best_relative_error": result.best_relative_error,
        "within_1_percent": result.best_relative_error < 0.01,
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

    @property
    def name(self) -> str:
        """Run name."""
        return str(self.result["name"])

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
    return LoadedRun(path=run_dir, config=config, result=result, trace=trace, params=params)


def write_benchmark_summary(runs: list[LoadedRun], out_dir: Path) -> tuple[Path, Path]:
    """Write ``benchmark.json`` and ``benchmark.md`` (a Markdown table) for a set of runs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "name": run.name,
            "ansatz": run.result["ansatz"],
            "optimizer": run.result["optimizer"],
            "backend": run.result["backend"],
            "energy": run.result["energy"],
            "best_energy": run.result["best_energy"],
            "exact_energy": run.result["exact_energy"],
            "relative_error": run.result["relative_error"],
            "best_relative_error": run.result["best_relative_error"],
            "nfev": run.result["nfev"],
            "wall_time_s": run.result["wall_time_s"],
            "path": str(run.path),
        }
        for run in runs
    ]
    json_path = out_dir / "benchmark.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    header = (
        "| run | ansatz | optimizer | backend | final E | best E | exact E | rel. err (best) "
        "| evaluations | time (s) |\n|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = "".join(
        f"| {r['name']} | {r['ansatz']} | {r['optimizer']} | {r['backend']} | "
        f"{r['energy']:.4f} | {r['best_energy']:.4f} | {r['exact_energy']:.4f} | "
        f"{100 * r['best_relative_error']:.3f} % | {r['nfev']} | {r['wall_time_s']:.1f} |\n"
        for r in rows
    )
    md_path = out_dir / "benchmark.md"
    md_path.write_text(header + body)
    return json_path, md_path
