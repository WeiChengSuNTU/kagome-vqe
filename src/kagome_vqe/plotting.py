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

__all__ = ["plot_convergence", "plot_ideal_vs_shots", "plot_runs"]

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

    Returns:
        The figure, or ``None`` if no run recorded an ideal trace.
    """
    groups: dict[str, list[LoadedRun]] = {}
    for run in runs:
        if run.ideal_trace is not None and len(run.ideal_trace) > 1:
            groups.setdefault(run.combo, []).append(run)
    if not groups:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for combo, group in groups.items():
        exact = float(group[0].result["exact_energy"])
        max_shots = max(float(r.ideal_trace["shots"].iloc[-1]) for r in group)  # type: ignore[index]
        min_shots = min(float(r.ideal_trace["shots"].iloc[1]) for r in group)  # type: ignore[index]
        grid = np.geomspace(max(min_shots, 1.0), max(max_shots, 2.0), 60)
        curves = []
        for run in group:
            trace = run.ideal_trace
            assert trace is not None
            err = np.maximum(trace["ideal_energy"].to_numpy(float) - exact, 1e-12)
            # error after spending s shots = error of the last iterate reached by s
            idx = np.searchsorted(trace["shots"].to_numpy(float), grid, side="right") - 1
            curves.append(err[np.clip(idx, 0, len(err) - 1)])
        stack = np.vstack(curves)
        (line,) = ax.plot(grid, np.median(stack, axis=0), lw=2, label=f"{combo} (n={len(group)})")
        if len(group) > 1:
            ax.fill_between(
                grid, stack.min(axis=0), stack.max(axis=0), color=line.get_color(), alpha=0.15
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Shots spent (all measurement bases)")
    ax.set_ylabel("Noise-free energy error of current parameters")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig
