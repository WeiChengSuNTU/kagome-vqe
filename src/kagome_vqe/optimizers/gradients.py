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
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

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
    """

    grad: FloatArray
    value: float | None
    nfev: int


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
    """Exact (shot-noise-limited) gradients from the two-term shift rule, per gate occurrence.

    For each of the ``K`` parameterised gates, evaluate the circuit with that gate's angle
    shifted by ``+pi/2`` and ``-pi/2``; half the difference is the derivative with respect to
    that occurrence. Occurrences are then summed per ansatz parameter
    (:meth:`ParameterMap.contract`). The objective must evaluate the *unshared* circuit, which
    :func:`kagome_vqe.vqe.prepare` compiles when ``optimizer.gradient = "parameter_shift"``.

    Args:
        parameter_map: Occurrence-to-parameter map of the compiled circuit. ``None`` means every
            parameter occurs once (then any objective works).
    """

    SHIFT = np.pi / 2

    def __init__(self, parameter_map: ParameterMap | None = None) -> None:
        """Create the estimator."""
        self.parameter_map = parameter_map

    @property
    def name(self) -> str:
        """``"parameter_shift"``."""
        return "parameter_shift"

    @property
    def stochastic(self) -> bool:
        """``False``."""
        return False

    def evaluations(self, num_parameters: int) -> int:
        """``2K``: two energies per gate occurrence."""
        k = num_parameters if self.parameter_map is None else self.parameter_map.num_occurrences
        return 2 * k

    def estimate(self, objective: Objective, x: FloatArray) -> GradientEstimate:
        """One batch of ``2K`` rows in circuit-parameter space."""
        mapping = self.parameter_map or ParameterMap.identity(x.size)
        phi = mapping.expand(x)
        shifts = self.SHIFT * np.eye(mapping.num_occurrences)
        rows = np.vstack([phi[None, :] + shifts, phi[None, :] - shifts])
        if isinstance(objective, ExpandableObjective):
            values = objective.batch_expanded(rows, base=x)
        elif mapping.is_identity:
            values = objective.batch(rows)
        else:
            msg = "parameter shift with shared parameters needs an ExpandableObjective"
            raise TypeError(msg)
        k = mapping.num_occurrences
        grad_phi = (values[:k] - values[k:]) / 2.0
        return GradientEstimate(mapping.contract(grad_phi), None, rows.shape[0])


def make_gradient(
    kind: str,
    *,
    fd_eps: float = 1e-6,
    perturbation: float = 0.1,
    rng: np.random.Generator | None = None,
    parameter_map: ParameterMap | None = None,
) -> GradientEstimator:
    """Build an estimator by name: ``forward``, ``central``, ``spsa`` or ``parameter_shift``."""
    factories: dict[str, Callable[[], GradientEstimator]] = {
        "forward": lambda: ForwardDifference(fd_eps),
        "central": lambda: CentralDifference(fd_eps),
        "spsa": lambda: SimultaneousPerturbation(perturbation, rng),
        "parameter_shift": lambda: ParameterShift(parameter_map),
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
