"""ADAM, with a pluggable gradient estimator.

The update rule mirrors the (now removed) ``qiskit.algorithms.optimizers.ADAM`` used by the
original notebook so its results can be reproduced: the same bias-corrected update and the same
defaults ``beta1 = 0.9``, ``beta2 = 0.99`` and ``noise_factor = 1e-8``. The default gradient
is the notebook's forward difference (``n + 1`` energies per iteration); any estimator from
:mod:`kagome_vqe.optimizers.gradients` can replace it, for example parameter shift, or SPSA's
two-point estimate for the SPSA-ADAM hybrid.

Reference: Kingma and Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015.
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
from kagome_vqe.optimizers.gradients import GradientEstimator, make_gradient

__all__ = ["Adam"]


class Adam:
    """Adaptive-moment gradient descent.

    Each iteration asks the gradient estimator for one batch of energies, updates the first
    moment ``m`` (smoothed gradient) and second moment ``v`` (smoothed squared gradient), and
    steps by ``lr * m_hat / (sqrt(v_hat) + eps)``. The normalisation makes every step about
    ``lr`` in size; independent noise in the gradient averages out in ``m``, but a consistent
    error does not.

    Args:
        maxiter: Iteration budget (the paper used 100).
        lr: Learning rate (the paper used 0.01).
        beta1: Exponential decay of the first moment.
        beta2: Exponential decay of the second moment.
        noise_factor: Added to the denominator for numerical stability.
        tol: Stop when the parameter update norm falls below this. Ignored for stochastic
            estimators such as simultaneous perturbation.
        fd_eps: Step for the finite-difference estimators. ``1e-6`` suits exact energies;
            under shot noise use something like ``0.1`` or the gradient is pure noise.
        gradient: ``"forward"`` (default), ``"central"``, or any :class:`GradientEstimator`
            instance (parameter shift, simultaneous perturbation, ...).
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
        gradient: str | GradientEstimator = "forward",
        *,
        amsgrad: bool = False,
    ) -> None:
        """See class docstring."""
        if maxiter < 1:
            msg = "maxiter must be >= 1"
            raise ValueError(msg)
        if isinstance(gradient, str) and gradient not in ("forward", "central"):
            msg = (
                f"gradient {gradient!r} must be 'forward', 'central' or a GradientEstimator "
                "instance (use make_gradient for 'spsa' and 'parameter_shift')"
            )
            raise ValueError(msg)
        self._gradient: GradientEstimator = (
            make_gradient(gradient, fd_eps=fd_eps) if isinstance(gradient, str) else gradient
        )
        self._maxiter = maxiter
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._noise = noise_factor
        self._tol = tol
        self._amsgrad = amsgrad

    @property
    def name(self) -> str:
        """``"adam"``, or ``"adam_<gradient>"`` for non-default estimators."""
        return "adam" if self._gradient.name == "forward" else f"adam_{self._gradient.name}"

    @property
    def maxiter(self) -> int:
        """Iteration budget."""
        return self._maxiter

    @property
    def gradient(self) -> GradientEstimator:
        """The gradient estimator."""
        return self._gradient

    def evaluations_per_iteration(self, num_parameters: int) -> int:
        """Whatever one gradient costs: ``n + 1``, ``2n + 1``, ``2`` or ``2K``."""
        return self._gradient.evaluations(num_parameters)

    def minimize(
        self,
        objective: Objective | Callable[[FloatArray], float],
        x0: FloatArray,
        *,
        callback: Callback | None = None,
    ) -> OptimizeResult:
        """Run ADAM from ``x0``.

        The returned ``fun`` is one extra evaluation at the final parameters.
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
            est = self._gradient.estimate(fn, x)
            nfev += est.nfev
            grad = est.grad
            history.append(np.nan if est.value is None else est.value)

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
                value = history[-1]
                callback(IterationInfo(nit=t, x=x.copy(), fun=value, nfev=nfev))
            # A random estimator can return an exact zero (direction orthogonal to the gradient),
            # so a small step only means convergence for deterministic estimators.
            if not self._gradient.stochastic and np.linalg.norm(step) < self._tol:
                message = f"update norm below tol={self._tol}"
                break

        final = fn(x)
        nfev += 1
        return OptimizeResult(
            x=x, fun=float(final), nfev=nfev, nit=t, success=True, message=message, history=history
        )

    def __repr__(self) -> str:
        """Debug representation."""
        return f"Adam(maxiter={self._maxiter}, lr={self._lr}, gradient={self._gradient.name!r})"
