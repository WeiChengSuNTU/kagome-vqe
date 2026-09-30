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

__all__ = ["plot_convergence", "plot_runs"]

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
