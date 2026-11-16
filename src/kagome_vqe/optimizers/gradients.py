"""Gradient estimators, separated from the update rules that consume them.

An optimiser step has two independent parts: estimating a gradient from energies, and turning
that gradient into a parameter update. This module provides the first part; :class:`~kagome_vqe.
optimizers.Adam` and :class:`~kagome_vqe.optimizers.GradientDescent` provide the second and
accept any estimator here. Every estimator asks for all the energies of one gradient as a
single batch, so one gradient is one job on hardware.

=====================  ========================  ==================================  ============
estimator              energies per gradient     bias                                noise
=====================  ========================  ==================================  ============
``ForwardDifference``  ``n + 1``                 ``O(h)``                            ``~ s / h``
``CentralDifference``  ``2n + 1``                ``O(h^2)``                          ``~ s / h``
``Simultaneous...``    ``2``                     right on average over directions    large
``ParameterShift``     ``2K`` (``K`` gates)      none                                ``~ s``
=====================  ========================  ==================================  ============

``s`` is the shot noise of one energy, ``h`` the finite-difference step, ``n`` the number of
ansatz parameters and ``K`` the number of parameterised gates (36 for HVA, 12 for RVB).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from kagome_vqe.ansatz.unshare import ParameterMap
from kagome_vqe.optimizers.base import FloatArray, Objective, as_objective

__all__ = [
    "CentralDifference",
    "ExpandableObjective",
    "ForwardDifference",
    "GradientEstimate",
    "GradientEstimator",
    "ParameterShift",
    "SimultaneousPerturbation",
    "estimate",
    "make_gradient",
]


@dataclass(frozen=True, slots=True)
class GradientEstimate:
    """One gradient estimate.

    Attributes:
        grad: Estimated gradient, shape ``(n,)``.
        value: Estimate of the cost at the current point, if the rows contained one.
        nfev: Energies spent.
        details: Estimator-specific extras, e.g. the per-noise-scale values and gradients of a
            zero-noise-extrapolated parameter-shift gradient (see :class:`ParameterShift`).
    """

    grad: FloatArray
    value: float | None
    nfev: int
    details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ExpandableObjective(Objective, Protocol):
    """An objective that can also evaluate circuit parameters ``phi`` directly.

    Needed by :class:`ParameterShift`, which shifts individual gate occurrences rather than
    whole ansatz parameters. :class:`kagome_vqe.vqe.EnergyObjective` implements it.
    """

    @property
    def parameter_map(self) -> ParameterMap:
        """Map from ansatz parameters to circuit parameters."""
        ...

    def batch_expanded(self, phis: FloatArray, *, base: FloatArray | None = None) -> FloatArray:
        """Costs at circuit parameters ``phis`` of shape ``(k, K)``.

        ``base`` is the ansatz-parameter point the rows were derived from, for bookkeeping.
        """
        ...


@runtime_checkable
class GradientEstimator(Protocol):
    """Estimate the gradient of an objective at a point from one batch of energies."""

    @property
    def name(self) -> str:
        """Short identifier, e.g. ``"parameter_shift"``."""
        ...

    @property
    def stochastic(self) -> bool:
        """Whether estimates are random even for exact energies (then a tiny step is no signal)."""
        ...

    def evaluations(self, num_parameters: int) -> int:
        """Energies one gradient costs for ``num_parameters`` ansatz parameters."""
        ...

    def estimate(self, objective: Objective, x: FloatArray) -> GradientEstimate:
        """Gradient at ``x``."""
        ...


class ForwardDifference:
    """``g_i = [E(x + h e_i) - E(x)] / h``; the first row is ``x`` itself."""

    def __init__(self, step: float = 1e-6) -> None:
        """Create the estimator with step ``h``."""
        if step <= 0:
            msg = "finite-difference step must be positive"
            raise ValueError(msg)
        self.step = step

    @property
    def name(self) -> str:
        """``"forward"``."""
        return "forward"

    @property
    def stochastic(self) -> bool:
        """``False``."""
        return False

    def evaluations(self, num_parameters: int) -> int:
        """``n + 1``."""
        return num_parameters + 1

    def estimate(self, objective: Objective, x: FloatArray) -> GradientEstimate:
        """One batch: ``x`` and its ``n`` forward shifts."""
        n = x.size
        rows = np.vstack([x[None, :], x[None, :] + self.step * np.eye(n)])
        values = objective.batch(rows)
        grad = (values[1:] - values[0]) / self.step
        return GradientEstimate(np.asarray(grad, dtype=float), float(values[0]), rows.shape[0])


class CentralDifference:
    """``g_i = [E(x + h e_i) - E(x - h e_i)] / 2h``; the first row is ``x`` for bookkeeping."""

    def __init__(self, step: float = 1e-6) -> None:
        """Create the estimator with step ``h``."""
        if step <= 0:
            msg = "finite-difference step must be positive"
            raise ValueError(msg)
        self.step = step

    @property
    def name(self) -> str:
        """``"central"``."""
        return "central"

    @property
    def stochastic(self) -> bool:
        """``False``."""
        return False

    def evaluations(self, num_parameters: int) -> int:
        """``2n + 1``."""
        return 2 * num_parameters + 1

    def estimate(self, objective: Objective, x: FloatArray) -> GradientEstimate:
        """One batch: ``x``, then forward and backward shifts."""
        n = x.size
        shifts = self.step * np.eye(n)
        rows = np.vstack([x[None, :], x[None, :] + shifts, x[None, :] - shifts])
        values = objective.batch(rows)
        grad = (values[1 : n + 1] - values[n + 1 :]) / (2 * self.step)
        return GradientEstimate(np.asarray(grad, dtype=float), float(values[0]), rows.shape[0])


class SimultaneousPerturbation:
    """SPSA's two-point estimate ``g = [E(x + c D) - E(x - c D)] / 2c * D``, ``D`` random +-1.

    Feeding this into ADAM gives the SPSA-ADAM hybrid: two energies per step, with ADAM's moment
    averaging smoothing the very noisy single-direction estimates.
    """

    def __init__(
        self, perturbation: float = 0.1, seed: int | np.random.Generator | None = None
    ) -> None:
        """Create the estimator with perturbation ``c``."""
        if perturbation <= 0:
            msg = "perturbation must be positive"
            raise ValueError(msg)
        self.perturbation = perturbation
        self._rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    @property
    def name(self) -> str:
        """``"spsa"``."""
        return "spsa"

    @property
    def stochastic(self) -> bool:
        """``True``."""
        return True

    def evaluations(self, num_parameters: int) -> int:
        """Always 2."""
        del num_parameters
        return 2

    def estimate(self, objective: Objective, x: FloatArray) -> GradientEstimate:
        """One batch of two rows."""
        delta = self._rng.choice(np.array([-1.0, 1.0]), size=x.size)
        c = self.perturbation
        values = objective.batch(np.vstack([x + c * delta, x - c * delta]))
        grad = (values[0] - values[1]) / (2 * c) * delta
        return GradientEstimate(np.asarray(grad, dtype=float), float(values.mean()), 2)


class ParameterShift:
    """Exact (shot-noise-limited) gradients from two-term shift rules, per gate occurrence.

    For each of the ``K`` parameterised gate occurrences, evaluate the circuit with that
    occurrence's angle shifted by ``+s`` and ``-s``; ``r [E(+s) - E(-s)]`` is the derivative
    with respect to that occurrence (``s = pi/2, r = 1/2`` for Pauli rotations; ``s = pi/4,
    r = 1`` for exchange gates, see :mod:`kagome_vqe.ansatz.unshare`). Occurrences are then
    summed per ansatz parameter (:meth:`ParameterMap.contract`). The objective must evaluate the
    *unshared* circuit, which :func:`kagome_vqe.vqe.prepare` compiles when
    ``optimizer.gradient = "parameter_shift"``.

    With ``include_value=True`` the unshifted point is evaluated in the same batch (one extra
    energy), so the cost at the current point is known: this is what a trajectory record needs
    (``C(theta_t)`` alongside ``g(theta_t)``).

    **Zero-noise extrapolation.** If the objective exposes the per-noise-scale energies of its
    last batch (``objective.last_scales``, set by :class:`~kagome_vqe.vqe.EnergyObjective` over
    a ZNE estimator), the gradient is the derivative of the *extrapolated* cost: per scale
    ``lambda`` the shift rule gives ``g_lambda``, and ``g_ZNE = sum_lambda w_lambda g_lambda``
    with ``w_lambda = dE_ZNE / dE_lambda`` evaluated at the current point
    (:func:`kagome_vqe.mitigation.zne_gradient_weights`). For polynomial fits ``w`` are the fit
    coefficients, so this equals extrapolating each gradient component; for the exponential fit
    it is the chain rule. Requires ``include_value=True`` (the weights need ``E_lambda`` at the
    current point). ``details`` then carries ``scales``, ``scale_values``, ``scale_stds``,
    ``scale_grads`` and ``weights``.

    Args:
        parameter_map: Occurrence-to-parameter map of the compiled circuit. ``None`` means every
            parameter occurs once as a Pauli rotation (then any objective works).
        include_value: Also evaluate the unshifted point (row 0 of the batch).
    """

    SHIFT = np.pi / 2

    def __init__(
        self, parameter_map: ParameterMap | None = None, *, include_value: bool = False
    ) -> None:
        """Create the estimator."""
        self.parameter_map = parameter_map
        self.include_value = include_value

    @property
    def name(self) -> str:
        """``"parameter_shift"``."""
        return "parameter_shift"

    @property
    def stochastic(self) -> bool:
        """``False``."""
        return False

    def evaluations(self, num_parameters: int) -> int:
        """``2K`` (``2K + 1`` with ``include_value``): two energies per gate occurrence."""
        k = num_parameters if self.parameter_map is None else self.parameter_map.num_occurrences
        return 2 * k + int(self.include_value)

    def estimate(self, objective: Objective, x: FloatArray) -> GradientEstimate:
        """One batch of ``2K`` (or ``2K + 1``) rows in circuit-parameter space."""
        mapping = self.parameter_map or ParameterMap.identity(x.size)
        phi = mapping.expand(x)
        k = mapping.num_occurrences
        shifts = mapping.occurrence_shifts[:, None] * np.eye(k)
        rows = np.vstack([phi[None, :] + shifts, phi[None, :] - shifts])
        if self.include_value:
            rows = np.vstack([phi[None, :], rows])
        if isinstance(objective, ExpandableObjective):
            values = objective.batch_expanded(rows, base=x)
        elif mapping.is_identity:
            values = objective.batch(rows)
        else:
            msg = "parameter shift with shared parameters needs an ExpandableObjective"
            raise TypeError(msg)
        offset = int(self.include_value)
        coefficients = mapping.occurrence_coefficients
        scales = getattr(objective, "last_scales", None)
        if scales is None or not self.include_value:
            plus, minus = values[offset : offset + k], values[offset + k :]
            grad = mapping.contract(coefficients * (plus - minus))
            value = float(values[0]) if self.include_value else None
            return GradientEstimate(grad, value, rows.shape[0])
        return self._extrapolated(mapping, coefficients, scales, float(values[0]), rows.shape[0])

    @staticmethod
    def _extrapolated(
        mapping: ParameterMap,
        coefficients: FloatArray,
        scales: dict[str, Any],
        value: float,
        nfev: int,
    ) -> GradientEstimate:
        from kagome_vqe.mitigation.wrappers import zne_gradient_weights

        k = mapping.num_occurrences
        energies = np.asarray(scales["energies"], dtype=float)  # (L, rows)
        stds = np.asarray(scales["stds"], dtype=float)
        factors = tuple(float(f) for f in scales["scales"])
        plus, minus = energies[:, 1 : 1 + k], energies[:, 1 + k :]
        scale_grads = np.vstack(
            [mapping.contract(coefficients * (p - m)) for p, m in zip(plus, minus, strict=True)]
        )
        weights = zne_gradient_weights(factors, energies[:, 0], stds[:, 0], scales["extrapolation"])
        grad = np.asarray(weights @ scale_grads, dtype=float)
        details = {
            "scales": np.asarray(factors, dtype=float),
            "scale_values": energies[:, 0].copy(),
            "scale_stds": stds[:, 0].copy(),
            "scale_grads": scale_grads,
            "weights": weights,
        }
        return GradientEstimate(grad, value, nfev, details)


def make_gradient(
    kind: str,
    *,
    fd_eps: float = 1e-6,
    perturbation: float = 0.1,
    rng: np.random.Generator | None = None,
    parameter_map: ParameterMap | None = None,
    include_value: bool = False,
) -> GradientEstimator:
    """Build an estimator by name: ``forward``, ``central``, ``spsa`` or ``parameter_shift``.

    ``include_value`` applies to ``parameter_shift`` only (finite differences and SPSA already
    report a value).
    """
    factories: dict[str, Callable[[], GradientEstimator]] = {
        "forward": lambda: ForwardDifference(fd_eps),
        "central": lambda: CentralDifference(fd_eps),
        "spsa": lambda: SimultaneousPerturbation(perturbation, rng),
        "parameter_shift": lambda: ParameterShift(parameter_map, include_value=include_value),
    }
    if kind not in factories:
        msg = f"unknown gradient estimator {kind!r}; choose from {sorted(factories)}"
        raise ValueError(msg)
    return factories[kind]()


def estimate(
    estimator: GradientEstimator, fn: Objective | Callable[[FloatArray], float], x: FloatArray
) -> GradientEstimate:
    """Convenience wrapper that accepts plain callables."""
    return estimator.estimate(as_objective(fn), np.asarray(x, dtype=float))
