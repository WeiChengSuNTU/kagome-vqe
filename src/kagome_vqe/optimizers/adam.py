"""ADAM with finite-difference gradients.

This mirrors the (now removed) ``qiskit.algorithms.optimizers.ADAM`` used by the original
notebook so its results can be reproduced: the same bias-corrected update, the same default
``beta1 = 0.9``, ``beta2 = 0.99`` and ``noise_factor = 1e-8``, and a forward-difference
gradient costing ``n + 1`` evaluations per iteration.

Reference: Kingma and Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

from kagome_vqe.optimizers.base import (
    Callback,
    FloatArray,
    IterationInfo,
    Objective,
    OptimizeResult,
    as_objective,
)

__all__ = ["Adam"]

GradientScheme = Literal["forward", "central"]


class Adam:
    """Adaptive-moment gradient descent with numerical gradients.

    Args:
        maxiter: Iteration budget (the paper used 100).
        lr: Learning rate (the paper used 0.01).
        beta1: Exponential decay of the first moment.
        beta2: Exponential decay of the second moment.
        noise_factor: Added to the denominator for numerical stability.
        tol: Stop when the parameter update norm falls below this.
        fd_eps: Finite-difference step. ``1e-6`` is fine for exact statevector energies; under
            shot noise use something like ``0.1`` or the gradient is pure noise.
        gradient: ``"forward"`` (``n + 1`` evaluations) or ``"central"`` (``2 n + 1``,
            more accurate).
        amsgrad: Use the AMSGrad variant (max of past second moments).
    """

    def __init__(
        self,
        maxiter: int = 100,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.99,
        noise_factor: float = 1e-8,
        tol: float = 1e-6,
        fd_eps: float = 1e-6,
        gradient: GradientScheme = "forward",
        *,
        amsgrad: bool = False,
    ) -> None:
        """See class docstring."""
        if maxiter < 1:
            msg = "maxiter must be >= 1"
            raise ValueError(msg)
        if gradient not in ("forward", "central"):
            msg = f"unknown gradient scheme {gradient!r}"
            raise ValueError(msg)
        self._maxiter = maxiter
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._noise = noise_factor
        self._tol = tol
        self._fd_eps = fd_eps
        self._gradient: GradientScheme = gradient
        self._amsgrad = amsgrad

    @property
    def name(self) -> str:
        """``"adam"``."""
        return "adam"

    @property
    def maxiter(self) -> int:
        """Iteration budget."""
        return self._maxiter

    def evaluations_per_iteration(self, num_parameters: int) -> int:
        """``n + 1`` for forward differences, ``2 n + 1`` for central."""
        return num_parameters + 1 if self._gradient == "forward" else 2 * num_parameters + 1

    def _gradient_batch(self, x: FloatArray) -> FloatArray:
        """Parameter points whose costs give the gradient (first row is ``x`` itself)."""
        n = x.size
        shifts = self._fd_eps * np.eye(n)
        if self._gradient == "forward":
            return np.vstack([x[None, :], x[None, :] + shifts])
        return np.vstack([x[None, :], x[None, :] + shifts, x[None, :] - shifts])

    def _gradient_from(self, values: FloatArray, n: int) -> tuple[float, FloatArray]:
        f0 = float(values[0])
        if self._gradient == "forward":
            grad = (values[1 : n + 1] - f0) / self._fd_eps
        else:
            grad = (values[1 : n + 1] - values[n + 1 : 2 * n + 1]) / (2 * self._fd_eps)
        return f0, np.asarray(grad, dtype=float)

    def minimize(
        self,
        objective: Objective | Callable[[FloatArray], float],
        x0: FloatArray,
        *,
        callback: Callback | None = None,
    ) -> OptimizeResult:
        """Run ADAM from ``x0``.

        Each iteration evaluates one batch (the point and its shifted copies), updates the
        moments and steps. The returned ``fun`` is the cost at the *final* parameters, which
        costs one extra evaluation only if the loop ran to ``maxiter``; otherwise the last
        gradient batch already contains it.
        """
        fn = as_objective(objective)
        x = np.array(x0, dtype=float)
        n = x.size
        m = np.zeros(n)
        v = np.zeros(n)
        v_hat = np.zeros(n)
        nfev = 0
        history: list[float] = []
        message = "maxiter reached"
        t = 0
        for t in range(1, self._maxiter + 1):
            values = fn.batch(self._gradient_batch(x))
            nfev += values.size
            f0, grad = self._gradient_from(values, n)
            history.append(f0)

            m = self._beta1 * m + (1 - self._beta1) * grad
            v = self._beta2 * v + (1 - self._beta2) * grad**2
            lr_eff = self._lr * np.sqrt(1 - self._beta2**t) / (1 - self._beta1**t)
            if self._amsgrad:
                v_hat = np.maximum(v_hat, v)
                step = lr_eff * m / (np.sqrt(v_hat) + self._noise)
            else:
                step = lr_eff * m / (np.sqrt(v) + self._noise)
            x = x - step

            if callback is not None:
                callback(IterationInfo(nit=t, x=x.copy(), fun=f0, nfev=nfev))
            if np.linalg.norm(step) < self._tol:
                message = f"update norm below tol={self._tol}"
                break

        final = fn(x)
        nfev += 1
        return OptimizeResult(
            x=x, fun=float(final), nfev=nfev, nit=t, success=True, message=message, history=history
        )

    def __repr__(self) -> str:
        """Debug representation."""
        return (
            f"Adam(maxiter={self._maxiter}, lr={self._lr}, fd_eps={self._fd_eps}, "
            f"gradient={self._gradient!r})"
        )
