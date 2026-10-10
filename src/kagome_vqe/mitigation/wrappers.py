"""Error mitigation that acts on energies: zero-noise extrapolation and twin-circuit rescaling.

Both wrap any :class:`~kagome_vqe.estimators.EnergyEstimator`, so they compose with each other
and with :class:`~kagome_vqe.mitigation.counts.CountsEnergy` (readout and symmetry), on Aer or
on IBM hardware.

**Zero-noise extrapolation (ZNE).** Measure the same energy with the two-qubit noise amplified
by factors ``lambda = 1, 3, 5`` (each two-qubit gate ``G`` becomes ``G (G^dag G)^k``) and fit
``E(lambda)`` back to ``lambda = 0``. The ``exponential`` model ``E = A exp(-b lambda)`` matches
the near-multiplicative damping measured in M0; ``linear``, ``quadratic`` and ``richardson``
are the usual polynomial alternatives. Cost: one evaluation per scale factor.

**Twin-circuit rescaling.** If noise multiplies every energy by roughly the same factor, that
factor can be measured on a *twin*: the same compiled circuit at other parameters whose
noise-free energy is known classically. ``E_mitigated = E_noisy * E_ideal(twin) /
E_noisy(twin)``. The twin is measured once, when the estimator is opened. The twin must not be
the state being estimated, or the test is circular; :func:`choose_twin` therefore excludes
ground states.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Sequence
from types import TracebackType
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import NDArray
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from kagome_vqe.estimators import EnergyBatch, EnergyEstimator, StatevectorEnergy
from kagome_vqe.transpile import IsaProblem, compile_for_backend, fold_two_qubit_gates

__all__ = [
    "Extrapolation",
    "TwinRescaledEnergy",
    "ZneEnergy",
    "choose_twin",
    "extrapolate",
]

log = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
Extrapolation = Literal["linear", "quadratic", "richardson", "exponential"]


def extrapolate(
    scales: Sequence[float], values: FloatArray, stds: FloatArray, method: Extrapolation
) -> tuple[float, float]:
    """Fit ``values(scales)`` and evaluate at zero noise; returns ``(estimate, standard error)``.

    Fits are weighted by ``1 / std**2`` when every std is positive. The standard error is
    propagated through the (linear) fit, for ``exponential`` through its logarithm.
    """
    x = np.asarray(scales, dtype=float)
    y = np.asarray(values, dtype=float)
    s = np.asarray(stds, dtype=float)
    weights = 1.0 / s if np.all(s > 0) else np.ones_like(s)
    if method == "exponential":
        if not (np.all(y < 0) or np.all(y > 0)):
            log.warning(
                "exponential extrapolation needs same-sign energies; falling back to linear"
            )
            return extrapolate(scales, values, stds, "linear")
        sign = float(np.sign(y[0]))
        log_y = np.log(np.abs(y))
        log_s = s / np.abs(y)
        value, err = extrapolate(scales, log_y, log_s, "linear")
        estimate = sign * float(np.exp(value))
        return estimate, abs(estimate) * err
    degree = {"linear": 1, "quadratic": 2, "richardson": len(x) - 1}[method]
    if len(x) <= degree:
        msg = f"{method} extrapolation needs more than {degree} scale factors"
        raise ValueError(msg)
    # Zero-noise value is a fixed linear combination sum_i c_i y_i of the measurements.
    vander = np.vander(x, degree + 1, increasing=True) * weights[:, None]
    coeffs = np.linalg.pinv(vander)[0] * weights
    estimate = float(coeffs @ y)
    err = float(np.sqrt(np.sum((coeffs * s) ** 2)))
    return estimate, err


class ZneEnergy:
    """Zero-noise extrapolation over locally folded copies of the circuit.

    Args:
        problem: ISA problem at noise scale 1.
        make_inner: Builds the estimator for one (folded) problem.
        scales: Odd noise-scale factors, starting with 1.
        extrapolation: Fit model, see :func:`extrapolate`.
    """

    def __init__(
        self,
        problem: IsaProblem,
        make_inner: Callable[[IsaProblem], EnergyEstimator],
        scales: Sequence[int] = (1, 3, 5),
        extrapolation: Extrapolation = "exponential",
    ) -> None:
        """Fold the circuit once per scale factor and build the inner estimators."""
        self._scales = tuple(int(s) for s in scales)
        self._method: Extrapolation = extrapolation
        self._inners = [
            make_inner(problem.with_circuit(fold_two_qubit_gates(problem.circuit, s)))
            for s in self._scales
        ]
        self._stack: contextlib.ExitStack | None = None

    @property
    def description(self) -> dict[str, Any]:
        """Settings plus the scale-1 inner estimator's description."""
        return {
            "kind": "zne",
            "scales": list(self._scales),
            "extrapolation": self._method,
            "inner": self._inners[0].description,
        }

    def __enter__(self) -> Self:
        """Open every inner estimator."""
        self._stack = contextlib.ExitStack()
        for inner in self._inners:
            self._stack.enter_context(inner)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close every inner estimator."""
        if self._stack is not None:
            self._stack.__exit__(exc_type, exc, tb)
            self._stack = None

    def evaluate(self, params: FloatArray, *, shots: int | None = None) -> EnergyBatch:
        """Evaluate at every scale and extrapolate each row to zero noise."""
        batches = [inner.evaluate(params, shots=shots) for inner in self._inners]
        values = np.vstack([b.energies for b in batches])
        stds = np.vstack([b.stds for b in batches])
        k = values.shape[1]
        energies = np.empty(k)
        errors = np.empty(k)
        for row in range(k):
            energies[row], errors[row] = extrapolate(
                self._scales, values[:, row], stds[:, row], self._method
            )
        return EnergyBatch(
            energies=energies,
            stds=errors,
            shots=batches[0].shots,
            job_id=batches[0].job_id,
            wall_time_s=sum(b.wall_time_s for b in batches),
        )


class TwinRescaledEnergy:
    """Divide out the damping factor measured on a twin with known noise-free energy.

    Args:
        inner: Estimator to correct (raw, readout-corrected or ZNE).
        twin_params: Circuit parameters of the twin, shape ``(P,)``.
        ideal_twin_energy: Noise-free energy of the twin, computed classically.
        twin_shots: Shots for the one-off twin measurement (``None``: the inner default).
    """

    def __init__(
        self,
        inner: EnergyEstimator,
        twin_params: FloatArray,
        ideal_twin_energy: float,
        twin_shots: int | None = None,
    ) -> None:
        """Store the twin; it is measured in ``__enter__``."""
        self._inner = inner
        self._twin = np.asarray(twin_params, dtype=float)
        self._ideal = float(ideal_twin_energy)
        self._twin_shots = twin_shots
        self._noisy: float | None = None
        self._noisy_std = 0.0

    @property
    def factor(self) -> float:
        """``E_ideal(twin) / E_noisy(twin)``."""
        if self._noisy is None:
            msg = "twin not measured yet; use the estimator as a context manager"
            raise RuntimeError(msg)
        return self._ideal / self._noisy

    @property
    def description(self) -> dict[str, Any]:
        """Twin, its energies and the inner description."""
        return {
            "kind": "twin_rescaling",
            "twin_params": self._twin.tolist(),
            "ideal_twin_energy": self._ideal,
            "noisy_twin_energy": self._noisy,
            "twin_shots": self._twin_shots,
            "inner": self._inner.description,
        }

    def __enter__(self) -> Self:
        """Open the inner estimator and measure the twin once."""
        self._inner.__enter__()
        batch = self._inner.evaluate(self._twin[None, :], shots=self._twin_shots)
        self._noisy = float(batch.energies[0])
        self._noisy_std = float(batch.stds[0])
        if abs(self._noisy) < 0.05 * abs(self._ideal):
            msg = f"twin energy {self._noisy:.3f} is too close to zero to rescale by"
            raise ValueError(msg)
        log.info("twin: ideal %.4f, noisy %.4f, factor %.4f", self._ideal, self._noisy, self.factor)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the inner estimator."""
        self._inner.__exit__(exc_type, exc, tb)

    def evaluate(self, params: FloatArray, *, shots: int | None = None) -> EnergyBatch:
        """Rescale the inner energies; the twin's own uncertainty is propagated."""
        batch = self._inner.evaluate(params, shots=shots)
        factor = self.factor
        assert self._noisy is not None
        rel_twin = self._noisy_std / abs(self._noisy)
        energies = batch.energies * factor
        stds = np.sqrt((factor * batch.stds) ** 2 + (energies * rel_twin) ** 2)
        return EnergyBatch(
            energies=energies,
            stds=stds,
            shots=batch.shots,
            job_id=batch.job_id,
            wall_time_s=batch.wall_time_s,
        )


def choose_twin(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    exact_energy: float,
    *,
    rng: np.random.Generator,
    step: float = np.pi / 4,
    max_points: int = 256,
) -> tuple[FloatArray, float]:
    """Pick twin parameters on a grid of multiples of ``step``, with the largest ``|E_ideal|``.

    With ``step = pi/2`` every Pauli rotation is Clifford, which keeps twin energies classically
    computable at any size. For HVA on the kagome cell that grid is useless: every point is
    either a ground state or has zero energy. The default ``pi/4`` grid finds informative twins
    but needs a classical simulation of the twin (a statevector here, fine up to ~20 qubits).
    Ground states are excluded, so a twin never coincides with the state the benchmark asks
    about. Small parameter counts are searched exhaustively, larger ones by random sampling.

    Returns:
        Twin parameters (same space as ``circuit``) and their noise-free energy.

    Raises:
        ValueError: If every candidate is a ground state or has (near) zero energy.
    """
    n = circuit.num_parameters
    levels = round(2 * np.pi / step)
    grid = np.arange(levels) * step
    if levels**n <= max_points:
        mesh = np.stack(np.meshgrid(*([grid] * n), indexing="ij"), axis=-1).reshape(-1, n)
    else:
        mesh = grid[rng.integers(0, levels, size=(max_points, n))]
    with StatevectorEnergy(compile_for_backend(circuit, hamiltonian, None)) as est:
        energies = est.evaluate(mesh).energies
    candidates = (np.abs(energies - exact_energy) > 1e-6) & (np.abs(energies) > 1e-6)
    if not np.any(candidates):
        msg = "no candidate twin with nonzero, non-ground energy; supply twin parameters"
        raise ValueError(msg)
    best = int(np.argmax(np.where(candidates, np.abs(energies), -np.inf)))
    return mesh[best].astype(float), float(energies[best])
