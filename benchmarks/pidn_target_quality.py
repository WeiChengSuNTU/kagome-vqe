"""Stage I target quality and PIDN open-loop prediction, per measurement pipeline.

Usage::

    uv run python benchmarks/pidn_target_quality.py RELABEL_DIR PIDN_CONFIG \
        [--reference LABEL=GLOB ...] [--only VARIANT ...] [--t-init 10]

``RELABEL_DIR`` is a ``kagome-vqe pidn-relabel`` directory (one run per pipeline and seed).
``--reference`` adds other gradient-descent runs under a label, e.g. the original ZNE runs
(``zne=results/<ts>-graph_hva_zne_gd/runs/*dep0.001_zne*``) or the noise-free ones. For each
pipeline: the quality of its Stage I targets against the exact noise-free gradient
(:func:`kagome_vqe.pidn.diagnostics.target_quality`) and the open-loop prediction quality of
the PIDN variants trained on them (:func:`~kagome_vqe.pidn.diagnostics.extrapolation_test`).
Writes ``<RELABEL_DIR>/summary/{targets,open_loop}.csv`` and ``report.md``.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kagome_vqe.pidn.diagnostics import extrapolation_test, load_trajectory, target_quality
from kagome_vqe.pidn.workflow import load_pidn_config


def _runs(relabel_dir: Path, references: list[str]) -> list[tuple[str, Path]]:
    runs = []
    for path in sorted((relabel_dir / "runs").glob("*")) if relabel_dir.name else []:
        if (path / "status.json").exists():
            result = json.loads((path / "result.json").read_text())
            runs.append((str(result["pipeline"]), path))
    for spec in references:
        label, pattern = spec.split("=", 1)
        runs.extend((label, Path(m)) for m in sorted(glob.glob(pattern)))
    return runs


def main() -> None:
    """Compute and write the tables."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("relabel_dir", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="Default: RELABEL_DIR/summary.")
    parser.add_argument("--reference", nargs="*", default=[])
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--t-init", type=int, default=10)
    args = parser.parse_args()
    cfg = load_pidn_config(args.config)
    variants = [v for v in cfg.variants if not args.only or v.label in args.only]
    targets, loops = [], []
    for label, path in _runs(args.relabel_dir, args.reference):
        traj = load_trajectory(path)
        executions = int(traj["circuit_executions"][args.t_init - 1])
        targets.append(
            {"pipeline": label, "run": path.name, "stage1_executions": executions}
            | target_quality(traj, t_init=args.t_init)
        )
        frame = extrapolation_test(traj, variants, t_init=args.t_init, horizon=20)
        frame.insert(0, "pipeline", label)
        frame.insert(1, "run", path.name)
        loops.append(frame)
    out = args.out or args.relabel_dir / "summary"
    out.mkdir(parents=True, exist_ok=True)
    target_frame = pd.DataFrame(targets)
    loop_frame = pd.concat(loops, ignore_index=True)
    target_frame.to_csv(out / "targets.csv", index=False)
    loop_frame.to_csv(out / "open_loop.csv", index=False)
    t = target_frame.groupby("pipeline", sort=False).mean(numeric_only=True)
    loop_frame["window"] = np.where(loop_frame["ahead"] <= 5, "1-5", "6-10")
    o = loop_frame.pivot_table(
        index="pipeline", columns=["variant", "window"], values="cosine_to_ideal", sort=False
    )
    lines = [
        f"# Stage I targets per pipeline: {args.relabel_dir.name}",
        "",
        "Means over seeds. `cos` = cosine of the measured Stage I gradient with the exact",
        "noise-free gradient (all 10 points / last 5); `abs(g - g_ideal)` = gradient error",
        "norm; `SNR late` = noise-free gradient norm / gradient error over the last 5 points;",
        "`cost err` = measured cost minus noise-free energy (mean, spread). Open loop: cosine",
        "of the surrogate's predicted gradient with the noise-free one, 1-5 and 6-10 steps",
        "after training.",
        "",
        "| pipeline | Stage I circuits | cos | cos late | abs(g - g_ideal) | SNR late | cost err |",
        "|---|---|---|---|---|---|---|",
    ]
    for pipeline, row in t.iterrows():
        lines.append(
            f"| {pipeline} | {row.stage1_executions:,.0f} | {row.cosine_to_ideal:.3f} | "
            f"{row.cosine_to_ideal_late:.3f} | {row.gradient_error:.3f} | "
            f"{row.signal_to_noise_late:.2f} | {row.value_error_mean:+.3f} "
            f"(+-{row.value_error_std:.3f}) |"
        )
    header = [" ".join(str(part) for part in col) for col in o.columns]
    lines += [
        "",
        "Open-loop cosine to the noise-free gradient:",
        "",
        "| pipeline | " + " | ".join(header) + " |",
        "|---|" + "---|" * len(header),
    ]
    lines += [
        f"| {pipeline} | " + " | ".join(f"{v:.3f}" for v in row) + " |"
        for pipeline, row in o.iterrows()
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
