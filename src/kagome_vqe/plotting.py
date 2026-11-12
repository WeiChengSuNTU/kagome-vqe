"""Matplotlib figures for convergence traces and benchmark comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")  # headless by default; callers may switch backends before importing
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from kagome_vqe.results import LoadedRun

__all__ = [
    "plot_convergence",
    "plot_ideal_vs_shots",
    "plot_noise_analysis",
    "plot_runs",
    "plot_study_figures",
]

Axis = Literal["evaluation", "iteration"]
Trace = Sequence[float] | NDArray[np.floating[Any]]


def plot_convergence(
    traces: Mapping[str, Trace],
    target: float | None,
    *,
    xlabel: str = "Iteration",
    title: str | None = None,
    out: Path | None = None,
) -> Figure:
    """Energy-versus-step curves for several runs with a dashed target line.

    Args:
        traces: ``{label: energies}``.
        target: Exact ground-state energy (drawn as a horizontal line), or ``None``.
        xlabel: Axis label (``"Iteration"`` or ``"Evaluation"``).
        title: Figure title.
        out: If given, the figure is saved there (format from the suffix).

    Returns:
        The figure (call ``plt.close(fig)`` when done in loops).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, energies in traces.items():
        ax.plot(range(1, len(energies) + 1), energies, lw=1.8, label=label)
    if target is not None:
        ax.axhline(target, color="tab:red", ls="--", lw=1.5, label=f"exact: {target:.4f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Energy")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def plot_runs(
    runs: Sequence[LoadedRun],
    *,
    axis: Axis = "iteration",
    out: Path | None = None,
    title: str | None = None,
) -> Figure:
    """Convenience: :func:`plot_convergence` for loaded result directories.

    With ``axis="iteration"`` the lowest energy of each optimiser iteration is shown, which
    puts ADAM (``n + 1`` evaluations per iteration) and SPSA (2) on a comparable footing. With
    ``axis="evaluation"`` every evaluation is a point, as in the original CSVs.
    """
    traces = {
        run.name: (run.energies_per_iteration() if axis == "iteration" else run.energies())
        for run in runs
    }
    targets = {float(run.result["exact_energy"]) for run in runs}
    target = targets.pop() if len(targets) == 1 else None
    return plot_convergence(
        traces,
        target,
        xlabel="Iteration" if axis == "iteration" else "Evaluation",
        title=title,
        out=out,
    )


def plot_ideal_vs_shots(
    runs: Sequence[LoadedRun],
    *,
    out: Path | None = None,
    title: str | None = None,
) -> Figure | None:
    """Noise-free energy error of the optimiser's parameters against shots spent.

    The fair comparison of optimisers and gradient estimators: the x-axis is the real cost
    (circuit shots over all measurement bases), the y-axis is how far the *parameters* are
    from the ground state, unaffected by the noise floor. Runs sharing a combo label (repeated
    seeds) are drawn as a median line with a min-max band, on a common log-spaced shot grid.
    Each ansatz gets its own panel.

    Returns:
        The figure, or ``None`` if no run recorded an ideal trace.
    """
    panels: dict[str, dict[str, list[LoadedRun]]] = {}
    for run in runs:
        if run.ideal_trace is not None and len(run.ideal_trace) > 1:
            ansatz = str(run.result.get("ansatz", "runs"))
            panels.setdefault(ansatz, {}).setdefault(run.combo, []).append(run)
    if not panels:
        return None
    fig, axes = plt.subplots(
        1, len(panels), figsize=(7 * len(panels), 5), squeeze=False, sharey=True
    )
    for ax, (ansatz, groups) in zip(axes[0], panels.items(), strict=True):
        for combo, group in groups.items():
            _ideal_band(ax, combo, group)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Shots spent (all measurement bases)")
        ax.set_title(ansatz.upper())
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("Noise-free energy error of current parameters")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def _ideal_band(ax: Any, combo: str, group: list[LoadedRun]) -> None:
    """Median and min-max band of the noise-free error over the runs of one combo."""
    traces = [r.ideal_trace for r in group if r.ideal_trace is not None]
    exact = float(group[0].result["exact_energy"])
    max_shots = max(float(t["shots"].iloc[-1]) for t in traces)
    min_shots = min(float(t["shots"].iloc[1]) for t in traces)
    grid = np.geomspace(max(min_shots, 1.0), max(max_shots, 2.0), 60)
    curves = []
    for trace in traces:
        err = np.maximum(trace["ideal_energy"].to_numpy(float) - exact, 1e-12)
        # error after spending s shots = error of the last iterate reached by s
        idx = np.searchsorted(trace["shots"].to_numpy(float), grid, side="right") - 1
        curves.append(err[np.clip(idx, 0, len(err) - 1)])
    stack = np.vstack(curves)
    label = combo.split("-", 1)[-1]
    (line,) = ax.plot(grid, np.median(stack, axis=0), lw=2, label=f"{label} (n={len(group)})")
    if len(group) > 1:
        ax.fill_between(
            grid, stack.min(axis=0), stack.max(axis=0), color=line.get_color(), alpha=0.12
        )


# -- multi-seed studies (kagome_vqe.study) ----------------------------------------------------


def _cost_band(
    ax: Any,
    label: str,
    traces: list[Any],
    exact: float,
    xcol: str,
    *,
    colour: Any = None,
) -> None:
    """Median (line) and min-max band (shade) of the noise-free relative error against cost.

    Each trace is a run's ``ideal_trace.csv``: the noise-free energy of the iterate after
    every iteration, with the evaluations (``nfev``) and shots spent so far. At cost ``x`` a
    run's error is that of the last iterate it had reached. Runs are put on a common
    log-spaced grid up to the shortest run's final cost, so the median never mixes finished
    and unfinished runs.
    """
    usable = [t for t in traces if t is not None and len(t) > 1 and float(t[xcol].iloc[-1]) > 0]
    if not usable:
        return
    first = max(1.0, min(float(t[xcol].iloc[1]) for t in usable))
    last = min(float(t[xcol].iloc[-1]) for t in usable)
    grid = np.geomspace(first, max(last, first * 1.01), 80)
    curves = []
    for trace in usable:
        err = np.maximum((trace["ideal_energy"].to_numpy(float) - exact) / abs(exact), 1e-12)
        idx = np.searchsorted(trace[xcol].to_numpy(float), grid, side="right") - 1
        curves.append(err[np.clip(idx, 0, len(err) - 1)])
    stack = np.vstack(curves)
    (line,) = ax.plot(
        grid, np.median(stack, axis=0), lw=2, color=colour, label=f"{label} (n={len(usable)})"
    )
    if len(usable) > 1:
        ax.fill_between(
            grid, stack.min(axis=0), stack.max(axis=0), color=line.get_color(), alpha=0.15
        )


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_study_figures(study_dir: Path, runs: Any, out_dir: Path) -> list[Path]:
    """Figures of a :mod:`kagome_vqe.study` from its saved runs; returns the files written.

    * ``error_vs_evaluations-<backend>.png``: noise-free relative error of the iterate versus
      energy evaluations (calibration included), one panel per (lattice, p, H0);
    * ``error_vs_shots-<backend>.png``: the same against total shots (shot-based backends);
    * ``fidelity_vs_depth-<lattice>.png``: ground-space fidelity of the final parameters versus
      p, mean and standard deviation over seeds;
    * ``seeds-<lattice>.png``: final noise-free relative error of every seed.
    """
    import pandas as pd

    study_dir = Path(study_dir)
    out_dir = Path(out_dir)
    written: list[Path] = []
    optimizers = sorted(runs["optimizer"].unique())
    colours = {opt: plt.get_cmap("tab10")(k % 10) for k, opt in enumerate(optimizers)}
    traces: dict[str, Any] = {}
    for path in runs["path"]:
        ideal = Path(path) / "ideal_trace.csv"
        traces[path] = pd.read_csv(ideal) if ideal.exists() else None
    for backend, frame in runs.groupby("backend", sort=True):
        exact_backend = bool(frame["total_shots"].max() == 0)
        for xcol, stem, xlabel in (
            ("nfev", "error_vs_evaluations", "Energy evaluations (incl. calibration)"),
            ("shots", "error_vs_shots", "Shots spent (all measurement bases)"),
        ):
            if xcol == "shots" and exact_backend:
                continue
            panels = list(frame.groupby(["lattice", "reps", "include_h0"], sort=True))
            fig, axes = plt.subplots(
                1, len(panels), figsize=(5.2 * len(panels), 4.2), squeeze=False, sharey=True
            )
            for ax, ((lattice, reps, h0), panel) in zip(axes[0], panels, strict=True):
                exact = float(panel["exact_energy"].iloc[0])
                for opt, group in panel.groupby("optimizer", sort=True):
                    _cost_band(
                        ax,
                        str(opt),
                        [traces[p] for p in group["path"]],
                        exact,
                        xcol,
                        colour=colours[opt],
                    )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel(xlabel)
                ax.set_title(f"{lattice}, p={reps}, H0 {'on' if h0 else 'off'}", fontsize=10)
                ax.grid(alpha=0.3, which="both")
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(fontsize=7)
            axes[0][0].set_ylabel("Noise-free relative energy error of iterate")
            fig.suptitle(
                f"graph_hva optimisers on {backend}\nmedian and min-max over seeds", fontsize=10
            )
            written.append(_save(fig, out_dir / f"{stem}-{backend}.png"))
    for lattice, frame in runs.groupby("lattice", sort=True):
        written.append(_fidelity_figure(str(lattice), frame, colours, out_dir))
        written.append(_seed_figure(str(lattice), frame, optimizers, colours, out_dir))
    del study_dir
    return written


def _fidelity_figure(lattice: str, frame: Any, colours: dict[str, Any], out_dir: Path) -> Path:
    """Ground-space fidelity of the final parameters versus p (mean +- std over seeds)."""
    backends = sorted(frame["backend"].unique())
    fig, axes = plt.subplots(
        1, len(backends), figsize=(5.2 * len(backends), 4.2), squeeze=False, sharey=True
    )
    for ax, backend in zip(axes[0], backends, strict=True):
        sub = frame[frame["backend"] == backend]
        for (opt, h0), group in sub.groupby(["optimizer", "include_h0"], sort=True):
            stats = group.groupby("reps")["fidelity"].agg(["mean", "std"]).fillna(0.0)
            ax.errorbar(
                stats.index.to_numpy() + (0.05 if h0 else 0.0),
                stats["mean"],
                yerr=stats["std"],
                marker="o" if not h0 else "s",
                ls="-" if not h0 else "--",
                capsize=3,
                color=colours[opt],
                label=f"{opt}, H0 {'on' if h0 else 'off'}",
            )
        ax.set_xlabel("HVA cycles p")
        ax.set_title(f"{lattice} on {backend}", fontsize=10)
        ax.set_xticks(sorted(sub["reps"].unique()))
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    axes[0][0].set_ylabel("Ground-space fidelity (mean +- std over seeds)")

    return _save(fig, out_dir / f"fidelity_vs_depth-{lattice}.png")


def _seed_figure(
    lattice: str, frame: Any, optimizers: list[str], colours: dict[str, Any], out_dir: Path
) -> Path:
    """Final noise-free relative error, one point per seed."""
    cells = list(frame.groupby(["backend", "reps", "include_h0"], sort=True))
    fig, ax = plt.subplots(figsize=(max(6.0, 1.6 * len(cells)), 4.2))
    width = 0.8 / max(1, len(optimizers))
    for k, opt in enumerate(optimizers):
        labelled = False
        for c, (_, cell) in enumerate(cells):
            points = cell[cell["optimizer"] == opt]["ideal_relative_error"].to_numpy(float)
            if points.size == 0:
                continue
            x = c - 0.4 + width * (k + 0.5)
            ax.scatter(
                np.full(points.size, x),
                np.maximum(points, 1e-12),
                color=colours[opt],
                s=22,
                label=None if labelled else opt,
            )
            labelled = True
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(
        [f"{b}\np={p}, H0 {'on' if h else 'off'}" for (b, p, h), _ in cells], fontsize=8
    )
    ax.set_yscale("log")
    ax.set_ylabel("Final noise-free relative energy error")
    ax.set_title(f"{lattice}: one point per seed")
    ax.grid(alpha=0.3, axis="y", which="both")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=7)
    return _save(fig, out_dir / f"seeds-{lattice}.png")


def plot_noise_analysis(points: Any, out_dir: Path) -> list[Path]:
    """Figures of a fixed-parameter noise analysis (:mod:`kagome_vqe.noise_analysis`).

    Per channel, against the error probability ``p_e`` (log-log), one line per source
    (optimiser, depth ``p``, seed):

    * ``infidelity-<channel>.png``: simulated infidelity ``1 - Tr(P_GS rho)`` (exact density
      matrix where available, else the trajectory mean with its 95 % CI) as markers, and the
      PRB 106 Eq. 9 estimate ``1 - (1 - p_e)^(n d) F0`` as a dashed line of the same colour;
    * ``energy_error-<channel>.png``: relative error of the noisy energy, with the noise-free
      error of the same parameters as a dotted horizontal line.
    """
    out_dir = Path(out_dir)
    written: list[Path] = []
    points = points[points["p"] > 0]
    for channel, frame in points.groupby("channel", sort=True):
        sources = list(frame.groupby(["source_optimizer", "reps", "seed"], sort=True))
        colours = {key: plt.get_cmap("tab10")(k % 10) for k, (key, _) in enumerate(sources)}
        fig_f, ax_f = plt.subplots(figsize=(6.4, 4.6))
        fig_e, ax_e = plt.subplots(figsize=(6.4, 4.6))
        for key, unsorted in sources:
            group = unsorted.sort_values("p")
            colour = colours[key]
            label = f"{key[0]} p={key[1]} s{key[2]}"
            ps = group["p"].to_numpy(dtype=float)
            if "exact_noisy_infidelity" in group and group["exact_noisy_infidelity"].notna().all():
                infid = group["exact_noisy_infidelity"].to_numpy(dtype=float)
                ax_f.plot(ps, infid, "o", color=colour, label=f"{label} (exact)")
            else:
                infid = group["traj_infidelity"].to_numpy(dtype=float)
                low = 1 - group["traj_fidelity_ci_high"].to_numpy(dtype=float)
                high = 1 - group["traj_fidelity_ci_low"].to_numpy(dtype=float)
                ax_f.errorbar(
                    ps,
                    infid,
                    yerr=[infid - low, high - infid],
                    fmt="s",
                    color=colour,
                    capsize=3,
                    label=f"{label} (trajectories)",
                )
            ax_f.plot(ps, group["eq9_infidelity"], "--", color=colour, alpha=0.8)
            col = (
                "exact_noisy_relative_error"
                if "exact_noisy_relative_error" in group
                and group["exact_noisy_relative_error"].notna().all()
                else "traj_relative_error"
            )
            ax_e.plot(ps, group[col], "o-", color=colour, label=label)
            ax_e.axhline(float(group["ideal_relative_error"].iloc[0]), color=colour, ls=":")
        for ax, ylabel in (
            (ax_f, "Infidelity 1 - F (dashed: PRB 106 Eq. 9)"),
            (ax_e, "Relative energy error (dotted: noise-free)"),
        ):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(f"Error probability p_e ({channel}, paper-layer noise)")
            ax.set_ylabel(ylabel)
            ax.grid(True, which="both", alpha=0.3)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7)
        written.append(_save(fig_f, out_dir / f"infidelity-{channel}.png"))
        written.append(_save(fig_e, out_dir / f"energy_error-{channel}.png"))
    return written
