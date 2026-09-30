"""Simultaneous perturbation stochastic approximation (SPSA).

SPSA estimates the gradient from just two evaluations per iteration regardless of the number of
parameters, which makes it the standard choice on shot-noisy hardware. With fixed
``learning_rate`` and ``perturbation`` it reproduces the original notebook's
``SPSA(maxiter=100, learning_rate=0.01, perturbation=0.01)``. With ``calibrate=True`` it uses
the Spall power-series gain sequences and the automatic calibration that
``qiskit.algorithms.optimizers.SPSA`` performed when no rates were given.

Reference: Spall, IEEE Trans. Automat. Control 37, 332 (1992).
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

__all__ = ["Spsa"]


class Spsa:
    """Two-evaluation stochastic gradient descent.

    Args:
        maxiter: Iteration budget.
        learning_rate: Step size ``a``. Ignored when ``calibrate`` is true.
        perturbation: Perturbation magnitude ``c``.
        calibrate: Use decaying gains ``a_k = a / (k + 1 + A)^alpha``,
            ``c_k = c / (k + 1)^gamma`` with ``a`` chosen from ``calibration_samples``
            gradient probes so that the first step has magnitude ``target_magnitude``.
        alpha: Exponent of the learning-rate schedule (Spall's 0.602).
        gamma: Exponent of the perturbation schedule (Spall's 0.101).
        stability_constant: ``A`` in the schedule; ``None`` means ``0.1 * maxiter``.
        calibration_samples: Gradient probes used by calibration (each costs 2 evaluations).
        target_magnitude: Desired first-step size for calibration (``2 pi / 10`` as in Qiskit).
        seed: Seed for the random perturbation directions. Pass a
            ``numpy.random.Generator`` to share a stream with the caller.
    """

    def __init__(
        self,
        maxiter: int = 100,
        learning_rate: float = 0.01,
        perturbation: float = 0.01,
        *,
        calibrate: bool = False,
        alpha: float = 0.602,
        gamma: float = 0.101,
        stability_constant: float | None = None,
        calibration_samples: int = 25,
        target_magnitude: float = 2 * np.pi / 10,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        """See class docstring."""
        if maxiter < 1:
            msg = "maxiter must be >= 1"
            raise ValueError(msg)
        if perturbation <= 0:
            msg = "perturbation must be positive"
            raise ValueError(msg)
        self._maxiter = maxiter
        self._a = learning_rate
        self._c = perturbation
        self._calibrate = calibrate
        self._alpha = alpha
        self._gamma = gamma
        self._A = 0.1 * maxiter if stability_constant is None else stability_constant
        self._cal_samples = calibration_samples
        self._target = target_magnitude
        self._rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    @property
    def name(self) -> str:
        """``"spsa"``."""
        return "spsa"

    @property
    def maxiter(self) -> int:
        """Iteration budget."""
        return self._maxiter

    def evaluations_per_iteration(self, num_parameters: int) -> int:
        """Always 2."""
        return 2

    def _direction(self, n: int) -> FloatArray:
        """A random ``+-1`` vector (Bernoulli, the standard SPSA choice)."""
        return self._rng.choice(np.array([-1.0, 1.0]), size=n)

    def _probe(
        self, fn: Objective, x: FloatArray, c: float
    ) -> tuple[FloatArray, float, float, FloatArray]:
        """Evaluate ``x +- c delta`` in one batch; return gradient estimate and both values."""
        delta = self._direction(x.size)
        values = fn.batch(np.vstack([x + c * delta, x - c * delta]))
        f_plus, f_minus = float(values[0]), float(values[1])
        grad = (f_plus - f_minus) / (2 * c) * delta  # 1/delta_i == delta_i for +-1
        return grad, f_plus, f_minus, delta

    def _calibrated_a(self, fn: Objective, x0: FloatArray) -> tuple[float, int]:
        """Qiskit-style calibration: pick ``a`` so the first step has ``target_magnitude``."""
        magnitudes = []
        for _ in range(self._cal_samples):
            grad, *_ = self._probe(fn, x0, self._c)
            magnitudes.append(float(np.linalg.norm(grad)))
        avg = float(np.mean(magnitudes))
        a = self._target / avg * (self._A + 1) ** self._alpha if avg > 0 else self._a
        return a, 2 * self._cal_samples

    def minimize(
        self,
        objective: Objective | Callable[[FloatArray], float],
        x0: FloatArray,
        *,
        callback: Callback | None = None,
    ) -> OptimizeResult:
        """Run SPSA from ``x0``.

        ``fun`` in the result is a genuine evaluation at the final point (one extra call), so
        the total is ``2 * maxiter + 1`` evaluations without calibration.
        """
        fn = as_objective(objective)
        x = np.array(x0, dtype=float)
        nfev = 0
        a = self._a
        if self._calibrate:
            a, used = self._calibrated_a(fn, x)
            nfev += used
        history: list[float] = []
        k = 0
        for k in range(1, self._maxiter + 1):
            if self._calibrate:
                a_k = a / (k + self._A) ** self._alpha
                c_k = self._c / k**self._gamma
            else:
                a_k, c_k = a, self._c
            grad, f_plus, f_minus, _ = self._probe(fn, x, c_k)
            nfev += 2
            x = x - a_k * grad
            estimate = 0.5 * (f_plus + f_minus)
            history.append(estimate)
            if callback is not None:
                callback(IterationInfo(nit=k, x=x.copy(), fun=estimate, nfev=nfev))
        final = fn(x)
        nfev += 1
        return OptimizeResult(
            x=x,
            fun=float(final),
            nfev=nfev,
            nit=k,
            success=True,
            message="maxiter reached",
            history=history,
        )

    def __repr__(self) -> str:
        """Debug representation."""
        return (
            f"Spsa(maxiter={self._maxiter}, learning_rate={self._a}, perturbation={self._c}, "
            f"calibrate={self._calibrate})"
        )
