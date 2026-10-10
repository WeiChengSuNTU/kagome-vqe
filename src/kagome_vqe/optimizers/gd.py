"""Plain gradient descent with a pluggable gradient estimator.

The simplest update rule, ``x <- x - lr * g``. Useful as a control when comparing gradient
estimators: with no moments, differences between runs come from the gradients alone.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from kagome_vqe.optimizers.base import (
    Callback,
    FloatArray,
    IterationInfo,
    Objective,
    OptimizeResult,
    as_objective,
)
from kagome_vqe.optimizers.gradients import GradientEstimator, ParameterShift

__all__ = ["GradientDescent"]


class GradientDescent:
    """``x <- x - lr * g`` for a fixed number of iterations.

    Args:
        maxiter: Iteration budget.
        lr: Step size.
        gradient: Any :class:`GradientEstimator`; defaults to exact parameter shift.
    """

    def __init__(
        self, maxiter: int = 100, lr: float = 0.05, gradient: GradientEstimator | None = None
    ) -> None:
        """See class docstring."""
        if maxiter < 1:
            msg = "maxiter must be >= 1"
            raise ValueError(msg)
        self._maxiter = maxiter
        self._lr = lr
        self._gradient: GradientEstimator = gradient or ParameterShift()

    @property
    def name(self) -> str:
        """``"gd_<gradient>"``."""
        return f"gd_{self._gradient.name}"

    @property
    def maxiter(self) -> int:
        """Iteration budget."""
        return self._maxiter

    def evaluations_per_iteration(self, num_parameters: int) -> int:
        """What one gradient costs."""
        return self._gradient.evaluations(num_parameters)

    def minimize(
        self,
        objective: Objective | Callable[[FloatArray], float],
        x0: FloatArray,
        *,
        callback: Callback | None = None,
    ) -> OptimizeResult:
        """Run gradient descent from ``x0``; ``fun`` is one extra evaluation at the end."""
        fn = as_objective(objective)
        x = np.array(x0, dtype=float)
        nfev = 0
        history: list[float] = []
        t = 0
        for t in range(1, self._maxiter + 1):
            est = self._gradient.estimate(fn, x)
            nfev += est.nfev
            history.append(np.nan if est.value is None else est.value)
            x = x - self._lr * est.grad
            if callback is not None:
                callback(IterationInfo(nit=t, x=x.copy(), fun=history[-1], nfev=nfev))
        final = fn(x)
        nfev += 1
        return OptimizeResult(
            x=x, fun=float(final), nfev=nfev, nit=t, message="maxiter reached", history=history
        )

    def __repr__(self) -> str:
        """Debug representation."""
        return (
            f"GradientDescent(maxiter={self._maxiter}, lr={self._lr}, "
            f"gradient={self._gradient.name!r})"
        )
