"""The original graph_hva depth sweep (L-BFGS-B, exact statevector), now with full persistence.

Usage (unchanged positional arguments)::

    uv run python benchmarks/graph_hva_vqe_sweep.py CASE {h0,noh0} SCHEDULE P1,P2,... \
        [--seeds 1,2] [--output-dir results] > out.jsonl

with ``CASE`` one of ``star``, ``bm2x6``, ``torus2x2``. As before, one JSON line per run is
written to stdout with the keys ``case, h0, schedule, p, seed, params, exact, energy, rel_err,
fidelity, nfev, nit, time_s``; a ``path`` key now points at the run's results directory.

What changed (2026-11): the sweep is a :mod:`kagome_vqe.study` like any other, so every run
is saved under ``<output-dir>/<timestamp>-graph_hva_sweep-<case>-<h0>/runs/`` with its
``config.toml``, ``result.json`` (initial and final parameters, resources, metadata),
``trace.csv`` and ``ideal_trace.csv``, and a summary is written next to them. The whole
36-run table is also available as one study: ``configs/studies/graph_hva_baseline.toml``.

The original sweep relied on scipy's default ``maxfun = 15000`` for L-BFGS-B, which stopped
every ``p >= 3`` run; that limit is now explicit so the old numbers are reproduced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from kagome_vqe.study import StudyConfig, StudyRun, run_study

CASES: dict[str, dict[str, Any]] = {
    "star": {"kind": "kagome_unit_cell"},
    "bm2x6": {"kind": "kagome_bm_rectangle", "rows": 2, "cols": 6},
    "torus2x2": {"kind": "kagome_periodic", "rows": 2, "cols": 2},
}
SEEDS = (1, 2)


def sweep_config(
    case: str, include_h0: bool, schedule: str, reps: list[int], seeds: list[int], output: Path
) -> StudyConfig:
    """The study equivalent of one invocation of the original script."""
    return StudyConfig.model_validate(
        {
            "name": f"graph_hva_sweep-{case}-{'h0' if include_h0 else 'noh0'}",
            "output_dir": str(output),
            "seeds": seeds,
            "x0_scale": 0.1,
            "reps": reps,
            "include_h0": [include_h0],
            "record_params": False,
            "lattices": [{"label": case, "lattice": CASES[case], "schedule": schedule}],
            "backends": [{"label": "sv", "backend": {"kind": "statevector"}}],
            "optimizers": [
                {
                    "label": "lbfgsb",
                    "optimizer": {
                        "type": "scipy",
                        "method": "L-BFGS-B",
                        "maxiter": 400,
                        "options": {"maxfun": 15000},
                    },
                }
            ],
        }
    )


def main(argv: list[str]) -> None:
    """Run the sweep described by the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("h0", choices=("h0", "noh0"))
    parser.add_argument("schedule")
    parser.add_argument("reps", help="comma-separated HVA cycles, e.g. 1,2,3")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    include_h0 = args.h0 == "h0"
    cfg = sweep_config(
        args.case,
        include_h0,
        args.schedule,
        [int(x) for x in args.reps.split(",")],
        [int(x) for x in args.seeds.split(",")],
        args.output_dir,
    )

    def emit(cell: StudyRun, result: Any, path: Path) -> None:
        row = {
            "case": args.case,
            "h0": include_h0,
            "schedule": args.schedule,
            "p": cell.reps,
            "seed": cell.seed,
            "params": int(result.optimal_params.size),
            "exact": result.exact_energy,
            "energy": result.final_energy,
            "rel_err": result.final_relative_error,
            "fidelity": result.ideal_fidelity,
            "nfev": result.nfev,
            "nit": result.nit,
            "time_s": result.wall_time_s,
            "path": str(path),
        }
        sys.stdout.write(json.dumps(row) + "\n")
        sys.stdout.flush()

    run_study(cfg, on_result=emit)


if __name__ == "__main__":
    main(sys.argv[1:])
