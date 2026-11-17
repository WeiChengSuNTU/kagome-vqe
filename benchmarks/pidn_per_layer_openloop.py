"""Test B: does per-layer parameter sharing (32 -> 10 angles) make PIDN's open loop work?

Usage::

    uv run python benchmarks/pidn_per_layer_openloop.py [--out DIR]

Same open-loop test as ``benchmarks/pidn_shot_scaling.py openloop`` (train on the first
``t_init = 10`` Stage I iterations of a recorded gradient-descent run, predict the gradient at
iterations 11-30 of the same path, cosine with the noise-free gradient; plus the memorisation
diagnostic), on

* ``per_layer_rs``: ``configs/studies/graph_hva_per_layer_gd.toml``, readout + symmetry,
  depolarising 1e-3 + readout 0.02, 1024 shots, 10 shared angles;
* ``per_layer_sv``: the same without noise (exact gradients, the clean limit);
* ``per_gate_rs``: the recorded 32-parameter runs of ``configs/studies/graph_hva_m2_gd.toml``
  (``full_rs``), i.e. the 1x value 0.15 of ``graph_hva_pidn_m2.md`` Sec. 3.5, recomputed
  here with the same code.

Variants: those of ``configs/pidn/graph_hva_pidn_capacity.toml`` (paper-sized, small, tiny
networks; quadratic and last-gradient baselines). Writes ``openloop.csv``, ``fit.csv``,
``targets.csv`` and ``report.md``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pidn_shot_scaling import evaluate, path_smoothness

from kagome_vqe.pidn.diagnostics import load_trajectory, target_quality
from kagome_vqe.pidn.workflow import load_pidn_config

log = logging.getLogger("per_layer_openloop")
ROOT = Path(__file__).resolve().parents[1]
T_INIT = 10
SOURCES = {
    "per_layer_rs": "results/*-graph_hva_per_layer_gd/runs/*-full_rs-gd_ps60-s[123]",
    "per_layer_sv": "results/*-graph_hva_per_layer_gd/runs/*-sv-gd_ps60-s[123]",
    "per_gate_rs": "results/*-graph_hva_m2_gd/runs/*-full_rs-gd_ps60-s[123]",
}
CONFIG = ROOT / "configs" / "pidn" / "graph_hva_pidn_capacity.toml"
VARIANTS = ("pidn_paper", "pidn_small", "pidn_tiny", "quadratic", "last_gradient")


def _completed(pattern: str) -> list[Path]:
    """Completed runs with a trajectory (an interrupted attempt is skipped)."""
    return [p for p in sorted(ROOT.glob(pattern)) if (p / "trajectory.npz").exists()]


def main() -> None:
    """Run every variant on every source and tabulate."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or ROOT / "results" / f"{stamp}-graph_hva_pidn_per_layer_openloop"
    out.mkdir(parents=True, exist_ok=True)
    variants = [v for v in load_pidn_config(CONFIG).variants if v.label in VARIANTS]
    loops, fits, quality = [], [], []
    for source, pattern in SOURCES.items():
        for path in _completed(pattern):
            traj = load_trajectory(path)
            end = min(T_INIT + 20, len(traj["value"]))
            info = {"source": source, "run": path.name, "parameters": traj["theta"].shape[1]}
            noise = traj["grad"] - traj["ideal_grad"]
            quality.append(
                info
                | target_quality(traj, t_init=T_INIT)
                | path_smoothness(traj, T_INIT, end)
                | {
                    "snr_stage1": float(
                        np.mean(
                            np.linalg.norm(traj["ideal_grad"][:T_INIT], axis=1)
                            / np.maximum(np.linalg.norm(noise[:T_INIT], axis=1), 1e-12)
                        )
                    ),
                    "noise_norm_stage1": float(np.mean(np.linalg.norm(noise[:T_INIT], axis=1))),
                }
            )
            for variant in variants:
                rows, fit = evaluate(traj, variant, t_init=T_INIT, end=end)
                loops += [info | r for r in rows]
                fits.append(info | fit)
                log.info(
                    "%s %s %s: open loop 1-5 %.3f",
                    source,
                    path.name[-2:],
                    variant.label,
                    np.mean([r["cosine_to_ideal"] for r in rows[:5]]),
                )
            pd.DataFrame(loops).to_csv(out / "openloop.csv", index=False)
            pd.DataFrame(fits).to_csv(out / "fit.csv", index=False)
            pd.DataFrame(quality).to_csv(out / "targets.csv", index=False)
    loop = pd.DataFrame(loops)
    ahead = (
        loop[loop["ahead"] <= 5].groupby(["source", "variant"])["cosine_to_ideal"].mean().unstack()
    )
    far = loop[loop["ahead"] > 5].groupby(["source", "variant"])["cosine_to_ideal"].mean().unstack()
    text = [
        "# Per-layer sharing: PIDN open loop",
        "",
        "Mean cosine with the noise-free gradient, 1-5 steps ahead:",
        "",
        ahead.round(3).to_string(),
        "",
        "6-20 steps ahead:",
        "",
        far.round(3).to_string(),
        "",
        "Stage I targets:",
        "",
        pd.DataFrame(quality).groupby("source").mean(numeric_only=True).round(3).T.to_string(),
    ]
    (out / "report.md").write_text("\n".join(text) + "\n")
    sys.stdout.write("\n".join(text) + "\n")


if __name__ == "__main__":
    main()
