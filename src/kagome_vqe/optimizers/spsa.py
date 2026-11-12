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
        max_step_norm: Optional step control: an update ``a_k g_k`` longer than this (Euclidean
            norm) is rescaled to this length. ``None`` (default): no control, the classic
            algorithm.
        clip_update: Optional step control: each update component is clipped to
            ``[-clip_update, clip_update]`` (applied before ``max_step_norm``). ``None``
            (default): no clipping.
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
        max_step_norm: float | None = None,
        clip_update: float | None = None,
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
        for value, what in ((max_step_norm, "max_step_norm"), (clip_update, "clip_update")):
            if value is not None and value <= 0:
                msg = f"{what} must be positive"
                raise ValueError(msg)
        self._max_step_norm = max_step_norm
        self._clip = clip_update
        self._rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self.step_norms: list[float] = []
        """Euclidean norm of every parameter update actually applied in the last run."""
        self.raw_step_norms: list[float] = []
        """Norm of every update before step control (equal to :attr:`step_norms` without it)."""
        self.controlled_steps = 0
        """Updates that step control shortened in the last run."""
        self.calibration_evaluations = 0
        """Evaluations the last :meth:`minimize` spent on calibration (0 without it)."""
        self.calibrated_learning_rate: float | None = None
        """The ``a`` calibration chose in the last :meth:`minimize`, if it ran."""

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

    @property
    def calibrate(self) -> bool:
        """Whether gains are calibrated."""
        return self._calibrate

    @property
    def calibration_cost(self) -> int:
        """Evaluations calibration will spend (``2 * calibration_samples``, or 0)."""
        return 2 * self._cal_samples if self._calibrate else 0

    def schedule(self) -> dict[str, float | int | bool | None]:
        """Gains and step control actually used (after calibration, if it ran)."""
        return {
            "calibrate": self._calibrate,
            "learning_rate": self.calibrated_learning_rate
            if self.calibrated_learning_rate is not None
            else self._a,
            "perturbation": self._c,
            "alpha": self._alpha if self._calibrate else None,
            "gamma": self._gamma if self._calibrate else None,
            "stability_constant": self._A if self._calibrate else None,
            "target_magnitude": self._target if self._calibrate else None,
            "calibration_samples": self._cal_samples if self._calibrate else 0,
            "max_step_norm": self._max_step_norm,
            "clip_update": self._clip,
        }

    def _control(self, step: FloatArray) -> FloatArray:
        """Apply the optional step control to an update ``a_k g_k``."""
        raw = float(np.linalg.norm(step))
        controlled = step
        if self._clip is not None:
            controlled = np.clip(controlled, -self._clip, self._clip)
        if self._max_step_norm is not None:
            norm = float(np.linalg.norm(controlled))
            if norm > self._max_step_norm:
                controlled = controlled * (self._max_step_norm / norm)
        applied = float(np.linalg.norm(controlled))
        self.raw_step_norms.append(raw)
        self.step_norms.append(applied)
        if applied < raw * (1 - 1e-12):
            self.controlled_steps += 1
        return controlled

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
            self.calibration_evaluations += 2  # kept current even if a budget stops us here
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
        self.calibration_evaluations = 0
        self.calibrated_learning_rate = None
        self.step_norms, self.raw_step_norms, self.controlled_steps = [], [], 0
        if self._calibrate:
            a, used = self._calibrated_a(fn, x)
            nfev += used
            self.calibrated_learning_rate = a
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
            x = x - self._control(a_k * grad)
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
            f"calibrate={self._calibrate}"
            + (f", max_step_norm={self._max_step_norm}" if self._max_step_norm is not None else "")
            + (f", clip_update={self._clip}" if self._clip is not None else "")
            + ")"
        )
