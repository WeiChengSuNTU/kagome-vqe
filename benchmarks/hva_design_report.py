"""Figure and table of the HVA design study (depth and H0 under PRB 106 layer noise).

Usage::

    uv run python benchmarks/hva_design_report.py POINTS_CSV [--out PNG]

``POINTS_CSV`` is the ``points.csv`` of ``kagome-vqe noise-analysis
configs/noise/graph_hva_design_fixed_noise.toml`` (a copy is in
``docs/experiments/data/graph_hva_design_fixed_noise_points.csv``). Medians over seeds of the
exact noisy ground-space fidelity and relative energy error, one line per ``(p, H0)``: colour =
``p``, solid circles = H0 off, dashed squares = H0 on. Prints the table of
``docs/research_overview.md``, Sec. 2.2.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLOURS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}  # categorical slots 1-4
INK, MUTED, GRID = "#1a1a19", "#5f5e58", "#e4e3de"


def main() -> None:
    """Plot and tabulate."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("points", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    points = pd.read_csv(args.points)
    med = (
        points.groupby(["reps", "include_h0", "p"])
        .agg(
            F0=("ideal_fidelity", "median"),
            err0=("ideal_relative_error", "median"),
            F=("exact_noisy_fidelity", "median"),
            err=("exact_noisy_relative_error", "median"),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    panels = (("F", "ground-space fidelity (exact, noisy)"), ("err", "relative energy error"))
    for ax, (column, label) in zip(axes, panels, strict=True):
        for _, group in med.groupby(["reps", "include_h0"]):
            reps, h0 = int(group["reps"].iloc[0]), bool(group["include_h0"].iloc[0])
            ax.plot(
                group["p"],
                group[column],
                color=COLOURS[reps],
                linestyle="--" if h0 else "-",
                marker="s" if h0 else "o",
                markersize=7,
                linewidth=2,
                label=f"p = {reps}, H0 {'on' if h0 else 'off'}",
            )
        ax.set_xscale("log")
        if column == "err":
            ax.set_yscale("log")
        ax.set_xlabel("depolarising error probability p_e (paper layers)", color=MUTED)
        ax.set_title(label, color=INK, fontsize=11, loc="left")
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(GRID)
    axes[1].legend(fontsize=8, frameon=False, ncol=2)
    fig.suptitle(
        "PRB 105 2x6 rectangle: noise-free optima under layer noise (median of 3 seeds)",
        color=INK,
        fontsize=11,
    )
    if args.out:
        fig.savefig(args.out, dpi=150)
    table = med.pivot_table(index=["reps", "include_h0"], columns="p", values=["F", "err"])
    sys.stdout.write(table.round(4).to_string() + "\n")


if __name__ == "__main__":
    main()
