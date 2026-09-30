"""Protocols and result types shared by all optimisers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Callback",
    "FunctionObjective",
    "IterationInfo",
    "Objective",
    "OptimizeResult",
    "Optimizer",
    "as_objective",
]

FloatArray = NDArray[np.float64]


@runtime_checkable
class Objective(Protocol):
    """A scalar cost function that can also evaluate many points at once."""

    def __call__(self, x: FloatArray) -> float:
        """Cost at a single point ``x`` of shape ``(n,)``."""
        ...

    def batch(self, xs: FloatArray) -> FloatArray:
        """Costs at ``xs`` of shape ``(k, n)``; returns shape ``(k,)``.

        Optimisers group the evaluations of one iteration into a single call so backends can
        submit them as one job.
        """
        ...


class FunctionObjective:
    """Adapt a plain ``f(x) -> float`` to :class:`Objective` by looping for batches."""

    def __init__(self, fn: Callable[[FloatArray], float]) -> None:
        """Wrap ``fn``."""
        self._fn = fn

    def __call__(self, x: FloatArray) -> float:
        """Evaluate one point."""
        return float(self._fn(np.asarray(x, dtype=float)))

    def batch(self, xs: FloatArray) -> FloatArray:
        """Evaluate each row in turn."""
        return np.asarray([self._fn(row) for row in np.asarray(xs, dtype=float)], dtype=float)


def as_objective(fn: Objective | Callable[[FloatArray], float]) -> Objective:
    """Return ``fn`` unchanged if it already supports batches, else wrap it."""
    if isinstance(fn, Objective):
        return fn
    return FunctionObjective(fn)


@dataclass(frozen=True, slots=True)
class IterationInfo:
    """What an optimiser reports to its callback after each iteration.

    Attributes:
        nit: Iteration number, starting at 1.
        x: Current parameter vector.
        fun: Cost at ``x`` as known to the optimiser (may be an estimate for SPSA).
        nfev: Cumulative objective evaluations so far.
    """

    nit: int
    x: FloatArray
    fun: float
    nfev: int


Callback = Callable[[IterationInfo], None]


@dataclass(slots=True)
class OptimizeResult:
    """Outcome of :meth:`Optimizer.minimize`.

    Attributes:
        x: Best parameters found.
        fun: Cost at ``x``.
        nfev: Total objective evaluations (single points; a batch of ``k`` counts ``k``).
        nit: Iterations performed.
        success: Whether the optimiser terminated normally.
        message: Human-readable termination reason.
        history: Cost per iteration, for convergence plots.
    """

    x: FloatArray
    fun: float
    nfev: int
    nit: int
    success: bool = True
    message: str = ""
    history: list[float] = field(default_factory=list)


@runtime_checkable
class Optimizer(Protocol):
    """Uniform interface over ADAM, SPSA and scipy."""

    @property
    def name(self) -> str:
        """Short identifier used in file names (e.g. ``"adam"``)."""
        ...

    @property
    def maxiter(self) -> int:
        """Iteration budget."""
        ...

    def evaluations_per_iteration(self, num_parameters: int) -> int | None:
        """Objective evaluations one iteration costs, or ``None`` if not fixed."""
        ...

    def minimize(
        self,
        objective: Objective | Callable[[FloatArray], float],
        x0: FloatArray,
        *,
        callback: Callback | None = None,
    ) -> OptimizeResult:
        """Minimise ``objective`` from ``x0``."""
        ...
