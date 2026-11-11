"""Run a graph_hva optimiser-study profile; a thin wrapper around ``kagome-vqe study``.

Usage::

    uv run python benchmarks/graph_hva_optimizer_study.py smoke            # ~1 minute
    uv run python benchmarks/graph_hva_optimizer_study.py representative   # ~25 minutes
    uv run python benchmarks/graph_hva_optimizer_study.py full --dry-run   # plan + estimate
    uv run python benchmarks/graph_hva_optimizer_study.py full --time-limit 60
    uv run python benchmarks/graph_hva_optimizer_study.py full --resume results/<dir> ...

A profile name maps to ``configs/studies/graph_hva_optimizer_<name>.toml``; any other first
argument is used as a config path. Remaining arguments go to ``kagome-vqe study`` unchanged.
All logic lives in :mod:`kagome_vqe.study`; this script only selects the file.
"""

from __future__ import annotations

import sys
from pathlib import Path

from kagome_vqe.cli import app

PROFILES = Path(__file__).resolve().parents[1] / "configs" / "studies"


def main(argv: list[str]) -> None:
    """Resolve the profile and hand over to the command line."""
    if not argv:
        sys.exit(__doc__)
    profile, *rest = argv
    path = PROFILES / f"graph_hva_optimizer_{profile}.toml"
    config = path if path.exists() else Path(profile)
    app(["study", str(config), *rest])


if __name__ == "__main__":
    main(sys.argv[1:])
