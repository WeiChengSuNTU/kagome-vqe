"""Reproduce the graph_hva VQE table in the README (exact statevector, two seeds per point).

Usage::

    uv run python benchmarks/graph_hva_vqe_sweep.py CASE {h0,noh0} SCHEDULE P1,P2,... > out.jsonl

with ``CASE`` one of ``star``, ``bm2x6``, ``torus2x2``. One JSON line per run is written to
stdout: lattice, H0, ``p``, seed, parameter count, exact and final energy, relative error,
ground-space fidelity, evaluations and wall time. The README used::

    star noh0 greedy 1        star h0 greedy 1
    bm2x6 noh0 prb105 1,2,3,4 bm2x6 h0 prb105 1,2,3,4
    torus2x2 noh0 greedy 1,2,3,4  torus2x2 h0 greedy 1,2,3,4

Each run is minutes to tens of minutes; set ``OMP_NUM_THREADS`` when running several at once.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

from kagome_vqe.config import RunConfig
from kagome_vqe.vqe import run_vqe

CASES: dict[str, dict[str, Any]] = {
    "star": {"kind": "kagome_unit_cell"},
    "bm2x6": {"kind": "kagome_bm_rectangle", "rows": 2, "cols": 6},
    "torus2x2": {"kind": "kagome_periodic", "rows": 2, "cols": 2},
}
SEEDS = (1, 2)


def main(argv: list[str]) -> None:
    """Run the sweep described by the command-line arguments."""
    case, h0_flag, schedule, reps_text = argv
    include_h0 = h0_flag == "h0"
    for p in (int(x) for x in reps_text.split(",")):
        for seed in SEEDS:
            cfg = RunConfig.model_validate(
                {
                    "name": f"{case}-p{p}",
                    "seed": seed,
                    "x0_scale": 0.1,
                    "lattice": CASES[case],
                    "ansatz": {
                        "type": "graph_hva",
                        "reps": p,
                        "include_h0": include_h0,
                        "schedule": schedule,
                    },
                    "optimizer": {"type": "scipy", "method": "L-BFGS-B", "maxiter": 400},
                    "backend": {"kind": "statevector"},
                    "record_params": False,
                }
            )
            start = time.time()
            result = run_vqe(cfg)
            row = {
                "case": case,
                "h0": include_h0,
                "schedule": schedule,
                "p": p,
                "seed": seed,
                "params": int(result.optimal_params.size),
                "exact": result.exact_energy,
                "energy": result.final_energy,
                "rel_err": result.final_relative_error,
                "fidelity": result.ideal_fidelity,
                "nfev": result.nfev,
                "nit": result.nit,
                "time_s": time.time() - start,
            }
            sys.stdout.write(json.dumps(row) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main(sys.argv[1:])
