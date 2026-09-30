"""Logging setup for the command line.

The library only ever calls ``logging.getLogger(__name__)``; this module is the one place that
installs handlers. Console output goes through Rich; a plain-text ``run.log`` is written into
each results directory via :func:`attach_file_handler`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler

__all__ = ["attach_file_handler", "configure_logging"]

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure_logging(level: int | str = logging.INFO, *, quiet_qiskit: bool = True) -> None:
    """Install a Rich console handler on the root logger (idempotent).

    Args:
        level: Root level.
        quiet_qiskit: Raise the Qiskit stack's loggers to WARNING; they are chatty at INFO.
    """
    root = logging.getLogger()
    if not any(isinstance(h, RichHandler) for h in root.handlers):
        handler = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        root.addHandler(handler)
    root.setLevel(level)
    if quiet_qiskit:
        for name in ("qiskit", "qiskit_aer", "qiskit_ibm_runtime", "urllib3", "matplotlib"):
            logging.getLogger(name).setLevel(logging.WARNING)


def attach_file_handler(path: Path, level: int = logging.DEBUG) -> logging.Handler:
    """Also write everything at ``level`` and above to ``path``; returns the handler."""
    handler = logging.FileHandler(Path(path), encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FORMAT))
    logging.getLogger().addHandler(handler)
    return handler
