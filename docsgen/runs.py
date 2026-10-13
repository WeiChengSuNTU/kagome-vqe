"""Read the benchmark summaries under ``results/`` into plain records.

Each ``results/<timestamp>-<name>-summary/benchmark.json`` is a list of per-run rows written by
``kagome_vqe.results.write_benchmark_summary``. Repeated seeds of one combination are named
``<combo>-s<k>``; :func:`aggregate` folds them into a mean and a sample standard deviation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np

from docsgen.paths import ROOT, summary_dir
from kagome_vqe.results import LoadedRun, load_run

__all__ = ["Aggregate", "aggregate", "by_name", "evaluation", "loaded_runs", "summary"]

_SEED_SUFFIX = re.compile(r"-s\d+$")


@cache
def summary(name: str) -> tuple[dict[str, Any], ...]:
    """Rows of ``results/*-<name>-summary/benchmark.json``."""
    rows: list[dict[str, Any]] = json.loads(
        (summary_dir(f"{name}-summary") / "benchmark.json").read_text()
    )
    return tuple(rows)


def by_name(name: str) -> dict[str, dict[str, Any]]:
    """Summary rows keyed by the run name with the summary prefix removed (``"hva_spsa"``)."""
    return {row["name"].split("-", 1)[1]: row for row in summary(name)}


@dataclass(frozen=True)
class Aggregate:
    """A metric over the seeds of one combination."""

    combo: str
    n: int
    mean: float
    std: float
    values: tuple[float, ...]


def aggregate(name: str, metric: str) -> dict[str, Aggregate]:
    """Mean and sample std of ``metric`` over seeds, keyed by combo (``"rvb_adam_forward"``)."""
    groups: dict[str, list[float]] = {}
    for row in summary(name):
        combo = _SEED_SUFFIX.sub("", row["name"].split("-", 1)[1])
        groups.setdefault(combo, []).append(float(row[metric]))
    out: dict[str, Aggregate] = {}
    for combo, values in groups.items():
        arr = np.asarray(values)
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        out[combo] = Aggregate(combo, arr.size, float(arr.mean()), std, tuple(values))
    return out


@cache
def loaded_runs(name: str) -> tuple[LoadedRun, ...]:
    """Full run directories (traces included) behind a summary."""
    return tuple(load_run(ROOT / row["path"]) for row in summary(name))


@cache
def evaluation(name: str) -> dict[str, dict[str, Any]]:
    """Rows of ``results/<name>/evaluation.json`` (fixed-point mode), keyed ``"hva_readout"``."""
    rows: list[dict[str, Any]] = json.loads((summary_dir(name) / "evaluation.json").read_text())
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row["below_ground"], bool):
            msg = f"{name}: below_ground must be a bool, got {row['below_ground']!r}"
            raise TypeError(msg)
        out[row["name"].split("-", 1)[1]] = row
    return out
