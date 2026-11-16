"""Summary of a PIDN study: tables, the comparison with the gradient-descent baselines, figures.

Every PIDN run continues a ZNE gradient-descent run of a study made with
``configs/studies/graph_hva_zne_gd_representative.toml``. That study also holds, per seed, the
baselines the paper compares with (unmitigated "noisy" gradient descent, ZNE gradient descent
itself) plus the noise-free and shots-only references, all with ``trajectory.npz``. The summary
puts PIDN and the baselines on the same axes: the *noise-free* relative energy error of the
current parameters against cumulative circuit executions (the paper's Figs. 5-6 use the ZNE
energy instead; the noise-free value is available in simulation and is the unbiased measure of
parameter quality), and the cosine between the gradient actually used and the exact noise-free
gradient (the paper's Fig. 4 uses the ZNE gradient as reference).

Speed-up as in the paper's Eq. 45: circuit executions ZNE gradient descent needs to reach a
target error divided by what PIDN needs, the target being the error ZNE gradient descent reaches
at the end of its budget (per seed). A surrogate run counts as reaching it only if it ends at or
below the target, at its first surrogate-phase point below it; otherwise "not reached" (``-``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kagome_vqe.trajectory_analysis import cosine, load_study_trajectories

__all__ = ["write_pidn_summary"]


def _first_reaching(executions: np.ndarray, errors: np.ndarray, target: float) -> float:
    hit = np.nonzero(errors <= target * (1 + 1e-12))[0]
    return float(executions[hit[0]]) if hit.size else math.nan


def _load_pidn_runs(study_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for path in sorted((study_dir / "runs").glob("*")):
        if not (path / "result.json").exists():
            continue
        result = json.loads((path / "result.json").read_text())
        with np.load(path / "trajectory.npz") as data:
            traj = {k: data[k] for k in data.files}
        runs.append({"path": path, "result": result, "trajectory": traj})
    return runs


def _baselines(sources: list[str]) -> list[Any]:
    """Trajectories of the gradient-descent study the sources belong to."""
    studies = {Path(s).parent.parent for s in sources}
    out = []
    for study in sorted(studies):
        if (study / "manifest.jsonl").exists():
            out.extend(load_study_trajectories(study))
    return out


def write_pidn_summary(study_dir: Path) -> Path:
    """(Re)write ``summary/`` of a PIDN study."""
    study_dir = Path(study_dir)
    out = study_dir / "summary"
    out.mkdir(exist_ok=True)
    runs = _load_pidn_runs(study_dir)
    sources_path = study_dir / "sources.json"
    sources = json.loads(sources_path.read_text()) if sources_path.exists() else []
    baselines = _baselines(sources)
    exact = runs[0]["result"]["exact_energy"] if runs else math.nan

    curves: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    targets: dict[int, tuple[float, float]] = {}
    for b in baselines:
        t = b.trajectory
        err = np.abs(t["ideal_energy"] - b.exact_energy) / abs(b.exact_energy)
        label = f"GD {b.labels['backend']}"
        final = b.final_ideal_energy
        final_err = math.nan if final is None else abs(final - b.exact_energy) / abs(b.exact_energy)
        curves.append(
            pd.DataFrame(
                {
                    "method": label,
                    "seed": b.labels["seed"],
                    "iteration": np.arange(len(err)),
                    "executions": t["circuit_executions"],
                    "error": err,
                    "cosine": cosine(t["grad"], t["ideal_grad"]),
                    "phase": 0,
                }
            )
        )
        rows.append(
            {
                "method": label,
                "seed": b.labels["seed"],
                "iterations": len(err),
                "circuit_executions": int(t["circuit_executions"][-1]),
                "final_error": final_err,
                "mean_cosine_late": float(np.nanmean(cosine(t["grad"], t["ideal_grad"])[10:]))
                if len(err) > 10
                else math.nan,
            }
        )
        if b.labels["backend"].endswith("_zne") and math.isfinite(final_err):
            # Executions at which ZNE-GD first reaches its own final error.
            all_err = np.append(err, final_err)
            all_exec = np.append(t["circuit_executions"], t["circuit_executions"][-1])
            targets[int(b.labels["seed"])] = (
                final_err,
                _first_reaching(all_exec, all_err, final_err),
            )
    for run in runs:
        res = run["result"]
        t = run["trajectory"]
        err = np.abs(t["ideal_energy"] - res["exact_energy"]) / abs(res["exact_energy"])
        phase = t["phase"]
        cos_used = cosine(t["grad"], t["ideal_grad"])
        cos_surr = cosine(t["surrogate_grad"], t["ideal_grad"])
        stage3 = phase > 0
        kind = res.get("variant_config", {}).get("model", {}).get("kind", "pidn")
        label = f"{'PIDN' if kind == 'pidn' else 'baseline'} {res['variant']}"
        curves.append(
            pd.DataFrame(
                {
                    "method": label,
                    "seed": res["seed"],
                    "iteration": np.arange(len(err)),
                    "executions": t["circuit_executions"],
                    "error": err,
                    "cosine": np.where(stage3, cos_surr, cos_used),
                    "phase": phase,
                }
            )
        )
        final_err = res.get("final_ideal_relative_error")
        target, zne_exec = targets.get(int(res["seed"]), (math.nan, math.nan))
        all_err = np.append(err, final_err if final_err is not None else np.nan)
        all_exec = np.append(t["circuit_executions"], t["circuit_executions"][-1])
        # Reached only if the run *ends* at or below the target, counted from the surrogate
        # phase on (Stage I is the ZNE run itself, which may dip below its own final error).
        t0 = int(res["t_init"])
        ends_below = final_err is not None and math.isfinite(target) and final_err <= target
        pidn_exec = _first_reaching(all_exec[t0:], all_err[t0:], target) if ends_below else math.nan
        r = res["resources"]
        rows.append(
            {
                "method": label,
                "seed": res["seed"],
                "iterations": len(err),
                "circuit_executions": r["circuit_executions"],
                "stage1_executions": r["stage1_circuit_executions"],
                "stage3_executions": r["stage3_circuit_executions"],
                "anchor_executions": r["anchor_circuit_executions"],
                "error_at_t_init": float(err[res["t_init"]])
                if len(err) > res["t_init"]
                else math.nan,
                "final_error": final_err,
                "final_zne_energy": res.get("final_zne_energy"),
                "final_unmitigated_energy": res.get("final_unmitigated_energy"),
                "mean_cosine_stage3": float(np.nanmean(cos_surr[stage3]))
                if stage3.any()
                else math.nan,
                "min_cosine_stage3": float(np.nanmin(cos_surr[stage3]))
                if stage3.any()
                else math.nan,
                "mean_anchor_cosine": float(np.nanmean(t["anchor_cosine"]))
                if np.isfinite(t["anchor_cosine"]).any()
                else math.nan,
                "zne_target_error": target,
                "zne_executions_to_target": zne_exec,
                "executions_to_target": pidn_exec,
                "speedup": zne_exec / pidn_exec
                if pidn_exec and math.isfinite(pidn_exec)
                else math.nan,
                "train_seconds": r["train_seconds"],
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(out / "runs.csv", index=False)
    curve_frame = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    curve_frame.to_csv(out / "curves.csv", index=False)
    report = _report(study_dir, table, exact)
    (out / "report.md").write_text(report)
    if not curve_frame.empty:
        _figures(curve_frame, runs, out / "figures")
    return out / "report.md"


def _fmt(value: Any, spec: str) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return format(value, spec)


def _report(study_dir: Path, table: pd.DataFrame, exact: float) -> str:
    lines = [
        f"# PIDN study {study_dir.name}",
        "",
        f"Noise-free relative energy error of the parameters (exact E0 = {exact:.6f}). Median over",
        "seeds, with the worst seed in brackets. `cos` is the mean cosine between the surrogate",
        "gradient and the exact noise-free gradient during the surrogate phase; for the",
        "gradient-descent baselines the mean over iterations >= 10 of the gradient actually used.",
        "Speed-up: circuit executions ZNE-GD needs to reach its own final error / executions",
        "the method needs (paper Eq. 45); `-` if the method does not end at or below it.",
        "",
        "| method | seeds | iterations | circuit executions | error at t_init | final error "
        "(worst) | cos | speed-up (per seed) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    if table.empty:
        return "\n".join(lines) + "\n"
    for method, sub in table.groupby("method", sort=False):
        surrogate = str(method).startswith(("PIDN", "baseline"))
        cos_col = "mean_cosine_stage3" if surrogate else "mean_cosine_late"
        speed = ", ".join(_fmt(v, ".2f") for v in sub["speedup"]) if "speedup" in sub else "-"
        at_init = sub["error_at_t_init"].median() if "error_at_t_init" in sub else math.nan
        lines.append(
            f"| {method} | {len(sub)} | {sub['iterations'].median():.0f} | "
            f"{sub['circuit_executions'].median():,.0f} | {_fmt(at_init, '.4f')} | "
            f"{_fmt(sub['final_error'].median(), '.4f')} ({_fmt(sub['final_error'].max(), '.4f')}) "
            f"| {_fmt(sub[cos_col].mean() if cos_col in sub else math.nan, '.3f')} | "
            f"{speed if surrogate else '-'} |"
        )
    return "\n".join(lines) + "\n"


_HEADLINE = ("pidn_paper", "pidn_anchor10", "pidn_nonoisy", "last_gradient", "quadratic")


def _error_figure(curves: pd.DataFrame, methods: list[str], path: Path) -> None:
    import matplotlib.pyplot as plt

    palette = plt.cm.tab20(np.linspace(0, 1, 20))
    colours = {m: palette[i % 20] for i, m in enumerate(methods)}
    seeds = sorted(curves["seed"].unique())
    surrogate_rows = curves["method"].str.startswith(("PIDN", "baseline"))
    by_cost = bool((curves.loc[surrogate_rows, "executions"] > 0).any())
    x_key = "executions" if by_cost else "iteration"
    fig, axes = plt.subplots(1, len(seeds), figsize=(5 * len(seeds), 4), sharey=True, squeeze=False)
    for ax, seed in zip(axes[0], seeds, strict=True):
        for method in methods:
            sub = curves[(curves["method"] == method) & (curves["seed"] == seed)]
            if sub.empty or (by_cost and not np.any(sub["executions"] > 0)):
                continue
            style = "-" if str(method).startswith("GD") else "--"
            ax.plot(sub[x_key], sub["error"], style, color=colours[method], lw=1.3, label=method)
        if by_cost:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("cumulative circuit executions" if by_cost else "iteration")
        ax.set_title(f"seed {seed}", fontsize=9)
        ax.grid(alpha=0.3, which="both")
    axes[0][0].set_ylabel("noise-free relative error")
    if axes[0][-1].get_legend_handles_labels()[0]:
        axes[0][-1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _figures(curves: pd.DataFrame, runs: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    methods = list(dict.fromkeys(curves["method"]))
    _error_figure(curves, methods, out_dir / "error_vs_executions.png")
    headline = [
        m for m in methods if m.startswith("GD") or m.split(" ", 1)[-1] in _HEADLINE
    ] or methods
    _error_figure(curves, headline, out_dir / "error_vs_executions_main.png")

    palette = plt.cm.tab20(np.linspace(0, 1, 20))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, method in enumerate(headline):
        sub = curves[curves["method"] == method]
        mean = sub.groupby("iteration")["cosine"].mean()
        ax.plot(mean.index, mean.values, color=palette[i % 20], lw=1.3, label=method)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("cos(gradient, noise-free gradient), seed mean")
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_dir / "gradient_cosine.png", dpi=150)
    plt.close(fig)
    _training_figure(runs, out_dir)


def _training_figure(runs: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    history = [r for r in runs if (r["path"] / "training.csv").exists()]
    if history:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for r in history[:6]:
            frame = pd.read_csv(r["path"] / "training.csv")
            first = frame[(frame["member"] == 0) & (frame["phase"] == 0)]
            name = r["result"]["variant"]
            ax.plot(first["epoch"], first["train_physics"], lw=1, label=f"{name} train phys")
            if "valid_physics" in first:
                ax.plot(
                    first["epoch"],
                    first["valid_physics"],
                    lw=1,
                    ls="--",
                    label=f"{name} held-out phys",
                )
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("normalised physics loss")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out_dir / "training_loss.png", dpi=150)
        plt.close(fig)
