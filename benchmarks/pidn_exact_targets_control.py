"""Control: PIDN open-loop prediction with *exact* targets on a recorded (noisy) path.

Usage::

    uv run python benchmarks/pidn_exact_targets_control.py PIDN_CONFIG RUN_DIR [RUN_DIR ...] \
        [--only VARIANT ...] [--t-init 10] [--out results/pidn_exact_targets.csv]

Replaces the recorded costs and gradients of each run's trajectory by the exact noise-free ones
(``ideal_energy``, ``ideal_grad``) and repeats the open-loop test. If prediction stays poor, the
limit is the geometry of the recorded path, not the noise of the training targets. Also prints,
per run, the noise-free gradient norm along the path and the cosine between consecutive
noise-free gradients (how fast the true gradient direction changes from step to step).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kagome_vqe.pidn.diagnostics import extrapolation_test, load_trajectory
from kagome_vqe.pidn.workflow import load_pidn_config


def main() -> None:
    """Run the control and write the rows."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--t-init", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("results/pidn_exact_targets.csv"))
    args = parser.parse_args()
    cfg = load_pidn_config(args.config)
    variants = [v for v in cfg.variants if not args.only or v.label in args.only]
    frames, lines = [], []
    for run in args.runs:
        traj = load_trajectory(run)
        exact = dict(traj)
        exact["grad"] = traj["ideal_grad"]
        exact["value"] = traj["ideal_energy"]
        exact["scale_values"] = traj["ideal_energy"][:, None]
        frame = extrapolation_test(exact, variants, t_init=args.t_init, horizon=2 * args.t_init)
        frame.insert(0, "run", run.name)
        frames.append(frame)
        g = traj["ideal_grad"][args.t_init - 1 : 2 * args.t_init]
        consecutive = np.sum(g[:-1] * g[1:], axis=1) / (
            np.linalg.norm(g[:-1], axis=1) * np.linalg.norm(g[1:], axis=1)
        )
        lines.append(
            f"{run.name}: noise-free |g| {np.linalg.norm(g, axis=1).mean():.2f}, "
            f"cos(g_t, g_t+1) {consecutive.mean():.3f}"
        )
    rows = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out, index=False)
    rows["window"] = np.where(rows["ahead"] <= 5, "1-5", "6-10")
    table = rows.pivot_table(index="variant", columns="window", values="cosine_to_ideal")
    sys.stdout.write("\n".join(lines) + "\n" + table.round(3).to_string() + f"\nwrote {args.out}\n")


if __name__ == "__main__":
    main()
