"""Where inputs live and where generated artefacts go."""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "CACHE",
    "REPORT_FIGURES",
    "REPORT_TABLES",
    "RESULTS",
    "ROOT",
    "SLIDES_DATA",
    "SLIDES_MEDIA",
    "summary_dir",
]

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
REPORT_FIGURES = ROOT / "report" / "figures"
REPORT_TABLES = ROOT / "report" / "tables"
SLIDES_DATA = ROOT / "slides" / "data"
SLIDES_MEDIA = ROOT / "slides" / "media"
CACHE = ROOT / "docsgen" / "cache"


def summary_dir(name: str) -> Path:
    """``results/<name>`` if it exists, else the newest ``results/<timestamp>-<name>``.

    Benchmark summaries are written without a timestamp; single runs are prefixed with a UTC
    timestamp, so the lexicographically last match is the most recent run of that name.

    Raises:
        FileNotFoundError: If no directory matches.
    """
    if (RESULTS / name).is_dir():
        return RESULTS / name
    matches = sorted(p for p in RESULTS.glob(f"*-{name}") if p.is_dir())
    if not matches:
        msg = f"no results directory ends with -{name} under {RESULTS}"
        raise FileNotFoundError(msg)
    return matches[-1]
