"""Plain gradient descent with a pluggable gradient estimator, optionally with momentum.

The simplest update rule, ``x <- x - lr * g``. Useful as a control when comparing gradient
estimators: with no moments, differences between runs come from the gradients alone. With
``momentum = beta > 0`` the heavy-ball rule ``v <- beta v + g``, ``x <- x - lr v`` averages the
gradients of the last ``~1 / (1 - beta)`` iterations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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

    Every iteration is kept in :attr:`records` (the point, the gradient, the value if the
    estimator measured it, cumulative evaluations and the estimator's ``details``), which is
    the gradient trajectory :func:`kagome_vqe.vqe.run_vqe` persists.

    Args:
        maxiter: Iteration budget.
        lr: Step size.
        gradient: Any :class:`GradientEstimator`; defaults to exact parameter shift.
        momentum: Heavy-ball coefficient ``beta`` in ``[0, 1)``; 0 is plain descent.
    """

    def __init__(
        self,
        maxiter: int = 100,
        lr: float = 0.05,
        gradient: GradientEstimator | None = None,
        *,
        momentum: float = 0.0,
    ) -> None:
        """See class docstring."""
        if maxiter < 1:
            msg = "maxiter must be >= 1"
            raise ValueError(msg)
        if not 0 <= momentum < 1:
            msg = f"momentum must be in [0, 1), got {momentum}"
            raise ValueError(msg)
        self._maxiter = maxiter
        self._lr = lr
        self._momentum = momentum
        self._gradient: GradientEstimator = gradient or ParameterShift()
        self.records: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        """``"gd_<gradient>"`` (``"gdm_<gradient>"`` with momentum)."""
        return f"{'gdm' if self._momentum else 'gd'}_{self._gradient.name}"

    @property
    def learning_rate(self) -> float:
        """Step size."""
        return self._lr

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
        self.records = []
        velocity = np.zeros_like(x)
        t = 0
        for t in range(1, self._maxiter + 1):
            est = self._gradient.estimate(fn, x)
            nfev += est.nfev
            history.append(np.nan if est.value is None else est.value)
            self.records.append(
                {
                    "theta": x.copy(),
                    "grad": np.array(est.grad, dtype=float),
                    "value": est.value,
                    "nfev": nfev,
                    "details": est.details,
                }
            )
            velocity = self._momentum * velocity + np.asarray(est.grad, dtype=float)
            x = x - self._lr * velocity
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
            f"gradient={self._gradient.name!r}, momentum={self._momentum})"
        )
