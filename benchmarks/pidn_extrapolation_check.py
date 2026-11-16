"""Open-loop extrapolation test of the PIDN variants on recorded trajectories.

Usage::

    uv run python benchmarks/pidn_extrapolation_check.py CONFIG RUN_DIR [RUN_DIR ...] \
        [--t-init 10] [--only LABEL ...] [--out results/pidn_extrapolation.csv]

``CONFIG`` is a PIDN study TOML (its variants are used; Stage III settings are ignored).
Each ``RUN_DIR`` is a gradient-descent run with ``trajectory.npz`` (noise-free or ZNE). The
variants are trained on the first ``t_init`` recorded steps and their gradients scored along
the recorded continuation (see :mod:`kagome_vqe.pidn.diagnostics`). Prints a summary table
and writes all rows to ``--out``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from kagome_vqe.pidn.diagnostics import extrapolation_test, load_trajectory
from kagome_vqe.pidn.workflow import load_pidn_config


def main() -> None:
    """Parse arguments, run the test, print and save the results."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--t-init", type=int, default=10)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=Path("results/pidn_extrapolation.csv"))
    args = parser.parse_args()
    cfg = load_pidn_config(args.config)
    variants = [v for v in cfg.variants if not args.only or v.label in args.only]
    frames = []
    for run in args.runs:
        frame = extrapolation_test(load_trajectory(run), variants, t_init=args.t_init)
        frame.insert(0, "run", run.name)
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out, index=False)
    bins = pd.cut(rows["ahead"], [0, 5, 20, 1000], labels=["1-5", "6-20", ">20"])
    table = rows.groupby(["variant", bins], observed=True, sort=False).agg(
        cos_ideal=("cosine_to_ideal", "mean"),
        cos_recorded=("cosine_to_recorded", "mean"),
        norm_ratio=("norm_ratio_to_ideal", "median"),
    )
    sys.stdout.write(table.round(3).to_string() + f"\nwrote {args.out}\n")


if __name__ == "__main__":
    main()
