"""Model gradient descent (Sung et al., Quantum Sci. Technol. 5, 044008 (2020)).

An online, noise-aware surrogate: instead of differentiating single noisy energies, every
iteration

1. samples ``k`` new points uniformly in the ball of radius ``delta_t`` around ``x_t`` and
   measures them (one batch);
2. collects *all* points measured so far that lie inside that ball (the trust region), so
   earlier samples keep contributing while the iterate moves slowly;
3. fits a full quadratic model ``q(x_t + d) = c + b.d + d^T A d / 2`` to them by least
   squares (minimum-norm solution when the region holds fewer points than the model has
   coefficients, as in the reference implementation ``recirq.optimize.mgd``);
4. steps ``x_{t+1} = x_t - gamma_t b``.

``gamma_t = lr / (t + 1 + A)^alpha`` and ``delta_t = delta / (t + 1)^xi`` (``A``, ``alpha``,
``xi`` default to 0: constant gains). Least squares averages the shot noise of many energies,
and the model is refitted at every step from the data near the current point; that is the
contrast to PIDN, which fits one surrogate to a finished trajectory and then extrapolates.
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

__all__ = ["ModelGradientDescent", "quadratic_model_gradient", "sample_ball"]


def sample_ball(rng: np.random.Generator, center: FloatArray, radius: float, k: int) -> FloatArray:
    """``k`` points uniformly distributed in the ball of ``radius`` around ``center``."""
    n = center.size
    directions = rng.normal(size=(k, n))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = radius * rng.uniform(size=(k, 1)) ** (1.0 / n)
    return center + radii * directions


def _features(d: FloatArray) -> FloatArray:
    """``[1, d_i, d_i d_j (i <= j)]`` per row of displacements ``d``."""
    n = d.shape[1]
    iu, ju = np.triu_indices(n)
    return np.hstack([np.ones((d.shape[0], 1)), d, d[:, iu] * d[:, ju]])


def quadratic_model_gradient(
    xs: FloatArray, ys: FloatArray, center: FloatArray
) -> tuple[float, FloatArray]:
    """Value and gradient at ``center`` of the least-squares quadratic through ``(xs, ys)``.

    The fit is centred at ``center`` so the gradient is the linear coefficient; with fewer
    points than coefficients the minimum-norm solution is used.
    """
    d = np.asarray(xs, dtype=float) - np.asarray(center, dtype=float)
    coef, *_ = np.linalg.lstsq(_features(d), np.asarray(ys, dtype=float), rcond=None)
    n = d.shape[1]
    return float(coef[0]), np.asarray(coef[1 : n + 1], dtype=float)


class ModelGradientDescent:
    """Model gradient descent; see the module docstring.

    Args:
        maxiter: Iteration budget.
        lr: Rate ``gamma``.
        sample_radius: ``delta``.
        sample_points: New energies per iteration ``k``.
        rate_decay_exponent: ``alpha``.
        stability_constant: ``A``.
        sample_radius_decay_exponent: ``xi``.
        seed: Seeds the sampling.
    """

    def __init__(
        self,
        maxiter: int = 100,
        lr: float = 0.05,
        sample_radius: float = 0.1,
        sample_points: int = 32,
        *,
        rate_decay_exponent: float = 0.0,
        stability_constant: float = 0.0,
        sample_radius_decay_exponent: float = 0.0,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        """See class docstring."""
        if maxiter < 1 or sample_points < 1 or sample_radius <= 0 or lr <= 0:
            msg = "maxiter, sample_points, sample_radius and lr must be positive"
            raise ValueError(msg)
        self._maxiter = maxiter
        self._lr = lr
        self._radius = sample_radius
        self._k = sample_points
        self._alpha = rate_decay_exponent
        self._A = stability_constant
        self._xi = sample_radius_decay_exponent
        self._rng = np.random.default_rng(seed)
        self.records: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        """``"mgd"``."""
        return "mgd"

    @property
    def maxiter(self) -> int:
        """Iteration budget."""
        return self._maxiter

    def evaluations_per_iteration(self, num_parameters: int) -> int:
        """``k`` new energies per iteration."""
        del num_parameters
        return self._k

    def minimize(
        self,
        objective: Objective | Callable[[FloatArray], float],
        x0: FloatArray,
        *,
        callback: Callback | None = None,
    ) -> OptimizeResult:
        """Run MGD from ``x0``; ``fun`` is one extra evaluation at the end."""
        fn = as_objective(objective)
        x = np.array(x0, dtype=float)
        xs = np.empty((0, x.size))
        ys = np.empty(0)
        nfev = 0
        history: list[float] = []
        self.records = []
        t = 0
        for t in range(1, self._maxiter + 1):
            radius = self._radius / t**self._xi
            rate = self._lr / (t + self._A) ** self._alpha
            new = sample_ball(self._rng, x, radius, self._k)
            values = np.asarray(fn.batch(new), dtype=float)
            nfev += self._k
            xs, ys = np.vstack([xs, new]), np.concatenate([ys, values])
            inside = np.linalg.norm(xs - x, axis=1) <= radius
            value, grad = quadratic_model_gradient(xs[inside], ys[inside], x)
            history.append(value)
            self.records.append(
                {
                    "theta": x.copy(),
                    "grad": grad,
                    "value": value,
                    "nfev": nfev,
                    "details": {"region_points": int(inside.sum()), "radius": radius},
                }
            )
            x = x - rate * grad
            if callback is not None:
                callback(IterationInfo(nit=t, x=x.copy(), fun=value, nfev=nfev))
        final = fn(x)
        nfev += 1
        return OptimizeResult(
            x=x, fun=float(final), nfev=nfev, nit=t, message="maxiter reached", history=history
        )

    def __repr__(self) -> str:
        """Debug representation."""
        return (
            f"ModelGradientDescent(maxiter={self._maxiter}, lr={self._lr}, "
            f"sample_radius={self._radius}, sample_points={self._k})"
        )
