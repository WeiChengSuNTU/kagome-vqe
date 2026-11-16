"""Gradient-trajectory analysis: how good are the gradients and costs an optimiser acted on?

Gradient-descent runs with ``optimizer.measure_current`` save ``trajectory.npz`` (see
:func:`kagome_vqe.vqe.gradient_trajectory`): per iteration the point ``theta_t``, the measured
cost (zero-noise extrapolated under ZNE), per-noise-scale values and gradients, the gradient
actually used, cumulative resources, and -- with ``track_ideal_gradient`` -- the exact
noise-free gradient and energy. This module turns a study's trajectories into

* per-iteration metrics: cosine similarity and relative error of the used gradient against
  the noise-free gradient, cost-estimation error ``value - E_ideal(theta_t)``, noise-free
  relative energy error, cumulative circuit executions;
* per-run summaries (means over iterations) and per-backend aggregates over seeds;
* figures: noise-free error versus circuit executions, gradient cosine versus iteration.

The same metrics are what a trajectory surrogate (PIDN) is judged by, so its runs can be
compared on equal terms.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "TrajectoryRun",
    "cosine",
    "load_study_trajectories",
    "summarise_trajectories",
    "trajectory_frame",
]

log = logging.getLogger(__name__)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity of two ``(T, n)`` arrays (NaN where a norm vanishes)."""
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norms > 0, np.sum(a * b, axis=1) / norms, np.nan)


@dataclass(slots=True)
class TrajectoryRun:
    """One run's trajectory with its labels."""

    labels: dict[str, Any]
    exact_energy: float
    trajectory: dict[str, np.ndarray]
    final_ideal_energy: float | None
    path: Path


def load_study_trajectories(study_dir: Path) -> list[TrajectoryRun]:
    """Every completed run of a study that saved ``trajectory.npz``."""
    from kagome_vqe.results import load_run
    from kagome_vqe.study import _read_manifest

    study_dir = Path(study_dir)
    runs = []
    latest: dict[str, dict[str, Any]] = {}
    for row in _read_manifest(study_dir):
        latest[row["name"]] = row
    for row in latest.values():
        if row.get("status") != "completed" or not row.get("path"):
            continue
        path = study_dir / row["path"]
        if not (path / "trajectory.npz").exists():
            continue
        run = load_run(path)
        assert run.trajectory is not None
        runs.append(
            TrajectoryRun(
                labels={k: row[k] for k in ("lattice", "reps", "backend", "optimizer", "seed")},
                exact_energy=float(run.result["exact_energy"]),
                trajectory=run.trajectory,
                final_ideal_energy=run.result.get("ideal_energy"),
                path=path,
            )
        )
    return runs


def trajectory_frame(run: TrajectoryRun) -> pd.DataFrame:
    """Per-iteration metrics of one run."""
    t = run.trajectory
    exact = run.exact_energy
    ideal_grad = t["ideal_grad"]
    grad = t["grad"]
    has_ideal = bool(np.all(np.isfinite(ideal_grad)))
    rows = pd.DataFrame(
        {
            "iteration": np.arange(len(t["value"])),
            "value": t["value"],
            "ideal_energy": t["ideal_energy"],
            "value_error": t["value"] - t["ideal_energy"],
            "ideal_relative_error": np.abs(t["ideal_energy"] - exact) / abs(exact),
            "grad_norm": np.linalg.norm(grad, axis=1),
            "ideal_grad_norm": np.linalg.norm(ideal_grad, axis=1) if has_ideal else np.nan,
            "cosine_to_ideal": cosine(grad, ideal_grad) if has_ideal else np.nan,
            "relative_grad_error": np.linalg.norm(grad - ideal_grad, axis=1)
            / np.linalg.norm(ideal_grad, axis=1)
            if has_ideal
            else np.nan,
            "circuit_executions": t["circuit_executions"],
            "shots": t["shots"],
        }
    )
    if t["scale_values"].shape[1] > 0:
        rows["unmitigated_value"] = t["scale_values"][:, 0]
        rows["unmitigated_error"] = t["scale_values"][:, 0] - t["ideal_energy"]
        rows["unmitigated_cosine"] = (
            cosine(t["scale_grads"][:, 0, :], ideal_grad) if has_ideal else np.nan
        )
    for key, value in run.labels.items():
        rows[key] = value
    return rows


def _run_summary(run: TrajectoryRun) -> dict[str, Any]:
    frame = trajectory_frame(run)
    exact = run.exact_energy
    final = run.final_ideal_energy
    return {
        **run.labels,
        "iterations": len(frame),
        "circuit_executions": int(frame["circuit_executions"].iloc[-1]),
        "shots": int(frame["shots"].iloc[-1]),
        "mean_cosine_to_ideal": float(frame["cosine_to_ideal"].mean()),
        "min_cosine_to_ideal": float(frame["cosine_to_ideal"].min()),
        "mean_relative_grad_error": float(frame["relative_grad_error"].mean()),
        "mean_value_error": float(frame["value_error"].mean()),
        "mean_abs_value_error": float(frame["value_error"].abs().mean()),
        "mean_unmitigated_error": float(frame.get("unmitigated_error", pd.Series(np.nan)).mean()),
        "final_ideal_relative_error": math.nan
        if final is None
        else abs(float(final) - exact) / abs(exact),
    }


def summarise_trajectories(
    runs: list[TrajectoryRun], out_dir: Path, *, title: str = "Gradient trajectories"
) -> Path:
    """Write ``trajectories.csv``, ``trajectory_runs.csv``, ``trajectories.md`` and figures."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not runs:
        msg = "no runs with trajectory.npz"
        raise FileNotFoundError(msg)
    per_iter = pd.concat([trajectory_frame(r) for r in runs], ignore_index=True)
    per_iter.to_csv(out_dir / "trajectories.csv", index=False)
    per_run = pd.DataFrame([_run_summary(r) for r in runs])
    per_run.to_csv(out_dir / "trajectory_runs.csv", index=False)
    keys = ["lattice", "reps", "backend", "optimizer"]
    grouped = per_run.groupby(keys, sort=False).agg(
        seeds=("seed", "count"),
        iterations=("iterations", "median"),
        executions=("circuit_executions", "median"),
        shots=("shots", "median"),
        cosine=("mean_cosine_to_ideal", "mean"),
        min_cosine=("min_cosine_to_ideal", "min"),
        grad_error=("mean_relative_grad_error", "mean"),
        value_error=("mean_value_error", "mean"),
        unmitigated_error=("mean_unmitigated_error", "mean"),
        final_error=("final_ideal_relative_error", "median"),
        final_error_max=("final_ideal_relative_error", "max"),
    )
    lines = [
        f"# {title}",
        "",
        "Means over iterations, then over seeds. `cos` and `grad err` compare the gradient the",
        "optimiser used with the exact noise-free gradient at the same point; `value err` is",
        "the measured cost (ZNE-extrapolated when mitigated) minus the noise-free energy of the",
        "same parameters; `raw err` is the same for the unamplified (scale 1) measurement.",
        "`final err` is the noise-free relative error of the final parameters (median, max).",
        "",
        "| p | backend | optimizer | seeds | iterations | circuit executions | shots | cos | "
        "min cos | grad err | value err | raw err | final err (median / max) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, row in grouped.iterrows():
        _, reps, backend, optimizer = (str(k) for k in key)  # type: ignore[attr-defined]
        lines.append(
            f"| {reps} | {backend} | {optimizer} | {int(row.seeds)} | {row.iterations:.0f} | "
            f"{row.executions:,.0f} | {row.shots:,.0f} | {row.cosine:.3f} | "
            f"{row.min_cosine:.3f} | {row.grad_error:.3f} | {row.value_error:+.4f} | "
            f"{row.unmitigated_error:+.4f} | {row.final_error:.4f} / {row.final_error_max:.4f} |"
        )
    (out_dir / "trajectories.md").write_text("\n".join(lines) + "\n")
    _plot(per_iter, out_dir)
    return out_dir / "trajectories.md"


def _plot(frame: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, ((backend, optimizer), sub) in enumerate(
        frame.groupby(["backend", "optimizer"], sort=False)
    ):
        colour = colours[i % len(colours)]
        label = f"{backend} / {optimizer}"
        for j, (_, run) in enumerate(sub.groupby("seed")):
            kw = {"color": colour, "alpha": 0.8, "lw": 1.2, "label": label if j == 0 else None}
            if np.any(run["circuit_executions"] > 0):
                axes[0].plot(run["circuit_executions"], run["ideal_relative_error"], **kw)
            axes[1].plot(run["iteration"], run["cosine_to_ideal"], **kw)
            axes[2].plot(run["iteration"], run["value_error"], **kw)
    axes[0].set_xlabel("cumulative circuit executions")
    axes[0].set_ylabel("noise-free relative error of theta_t")
    axes[0].set_yscale("log")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("cos(used gradient, noise-free gradient)")
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("measured cost - noise-free energy")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "trajectories.png", dpi=150)
    plt.close(fig)
