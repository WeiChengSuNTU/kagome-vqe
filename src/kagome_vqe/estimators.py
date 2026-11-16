"""Energy evaluation on simulators, fake backends and IBM Quantum.

Every backend family is wrapped in an :class:`EnergyEstimator` that takes a batch of parameter
vectors and returns a batch of energies. All implementations submit the batch as **one**
primitive unified bloc (PUB) with broadcast parameters, so an optimiser iteration is one job,
whatever the backend.

Estimators are context managers because the runtime ones own a ``Session``/``Batch``::

    with make_estimator(cfg.backend, problem, backend, seed=cfg.seed) as estimator:
        energies = estimator.evaluate(params).energies

Two properties matter for honest noisy benchmarks:

* **Shots are real.** Every shot-based estimator samples measurement outcomes, so shot noise
  and readout error are present exactly as on hardware. (Aer's own ``EstimatorV2`` computes
  expectation values without sampling; it is deliberately not used.)
* **Seeds are fresh per call.** A fixed simulator seed makes each row of a batch carry the same
  noise realisation on every call, which finite-difference optimisers read as a slope. Each
  estimator owns a seed *stream* derived from the run seed, so runs stay reproducible while
  successive calls are statistically independent.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import math
import time
from dataclasses import dataclass
from functools import cache
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from kagome_vqe.config import BackendConfig, MitigationConfig, Settings
from kagome_vqe.transpile import IsaProblem

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit

__all__ = [
    "DENSITY_MATRIX_MAX_QUBITS",
    "AerEnergy",
    "EnergyBatch",
    "EnergyEstimator",
    "LogicalNoiseEnergy",
    "RuntimeEnergy",
    "RuntimeSession",
    "StatevectorEnergy",
    "aer_noise_model",
    "make_estimator",
]

log = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

AerMethod = Literal["auto", "density_matrix", "statevector", "matrix_product_state"]

DENSITY_MATRIX_MAX_QUBITS = 12
"""Largest number of *active* qubits for which ``aer_method="auto"`` picks density matrices.

A density matrix on 12 qubits takes 256 MB and its cost does not depend on the shot count,
which makes large shot budgets free. Beyond that, ``auto`` falls back to statevector
trajectories, whose cost grows with the number of shots.
"""


@dataclass(frozen=True, slots=True)
class EnergyBatch:
    """Energies for a batch of parameter vectors.

    Attributes:
        energies: Shape ``(k,)``.
        stds: Standard errors as reported by the primitive, shape ``(k,)``; zeros for exact
            evaluation. For the sampling estimators this is the primitive's conservative bound
            ``sum_i |c_i| sigma_i / sqrt(shots)``, typically several times the real spread.
        shots: Shots per measurement basis, if known.
        job_id: Runtime job id, if any.
        wall_time_s: Seconds spent in the primitive call (including queueing).
        scales: For a zero-noise-extrapolated batch, the per-noise-scale raw energies before
            extrapolation: ``{"scales", "energies" (L, k), "stds" (L, k), "extrapolation"}``.
    """

    energies: FloatArray
    stds: FloatArray
    shots: int | None = None
    job_id: str | None = None
    wall_time_s: float = 0.0
    scales: dict[str, Any] | None = None


@runtime_checkable
class EnergyEstimator(Protocol):
    """Evaluate ``<psi(theta)| H |psi(theta)>`` for many ``theta`` at once."""

    @property
    def description(self) -> dict[str, Any]:
        """Backend and options, for the results metadata."""
        ...

    def evaluate(self, params: FloatArray, *, shots: int | None = None) -> EnergyBatch:
        """``params`` has shape ``(k, n)``; returns ``k`` energies.

        ``shots`` overrides the configured shot count for this call only (used for the final
        re-measurement); exact estimators ignore it.
        """
        ...

    def __enter__(self) -> Self:
        """Acquire resources (e.g. open a runtime session)."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Release resources."""
        ...


def _as_2d(params: FloatArray) -> FloatArray:
    arr = np.asarray(params, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        msg = f"params must have shape (k, n), got {arr.shape}"
        raise ValueError(msg)
    return arr


def _precision(shots: int | None) -> float | None:
    """Target precision that makes a V2 primitive run exactly ``shots`` shots.

    Primitives compute ``ceil(1 / precision**2)``. With ``precision = 1 / sqrt(shots)`` that
    can land one shot high through rounding (``1 / (1 / sqrt(128))**2 = 128.00000000000003``),
    so the precision is nudged up by a relative ``1e-12``.
    """
    return None if shots is None else (1.0 + 1e-12) / math.sqrt(shots)


def _job_result(job: Any, timeout_s: int) -> Any:
    """``job.result(timeout=...)`` where supported (cloud jobs), plain ``result()`` otherwise."""
    if "timeout" in inspect.signature(job.result).parameters:
        return job.result(timeout=timeout_s)
    return job.result()


@cache
def _transient_errors() -> tuple[type[BaseException], ...]:
    """Exceptions worth retrying: runtime/job failures, timeouts and network errors.

    Anything else (``TypeError``, ``ValueError``, ...) is a programming error and propagates
    immediately instead of being retried.
    """
    from qiskit_ibm_runtime import exceptions as rt

    return (
        rt.RuntimeJobFailureError,
        rt.RuntimeJobMaxTimeoutError,
        rt.RuntimeJobTimeoutError,
        rt.RuntimeInvalidStateError,
        rt.IBMRuntimeError,
        rt.IBMApiError,
        rt.JobError,
        rt.JobTimeoutError,
        ConnectionError,
        TimeoutError,
    )


def _shots_from_metadata(metadata: dict[str, Any]) -> int | None:
    for key in ("shots", "target_shots"):
        value = metadata.get(key)
        if isinstance(value, int | float) and value > 0:
            return int(value)
    return None


def _active_qubits(problem: IsaProblem) -> int:
    """Qubits touched by at least one instruction (Aer simulates only these)."""
    circuit = problem.circuit
    used = {circuit.find_bit(q).index for inst in circuit.data for q in inst.qubits}
    return len(used)


class _PrimitiveEnergy:
    """Shared logic: seed stream, one PUB per call, result unpacking."""

    def __init__(
        self,
        problem: IsaProblem,
        description: dict[str, Any],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self._problem = problem
        self._description = description
        self._configured_shots = shots
        self._seeds = np.random.default_rng(seed)

    @property
    def description(self) -> dict[str, Any]:
        """Backend and options."""
        return dict(self._description)

    def _next_seed(self) -> int:
        """A fresh simulator seed, drawn from the run's reproducible stream."""
        return int(self._seeds.integers(0, 2**31 - 1))

    def _unpack(
        self, job: Any, result: Any, params: FloatArray, elapsed: float, shots: int | None
    ) -> EnergyBatch:
        data = result[0].data
        energies = np.atleast_1d(np.asarray(data.evs, dtype=float))
        stds = np.atleast_1d(
            np.asarray(getattr(data, "stds", np.zeros_like(energies)), dtype=float)
        )
        if energies.shape != (params.shape[0],):
            msg = f"expected {params.shape[0]} energies, got shape {energies.shape}"
            raise RuntimeError(msg)
        metadata = dict(getattr(result[0], "metadata", {}) or {})
        job_id = getattr(job, "job_id", None)
        job_id_str = job_id() if callable(job_id) else job_id
        return EnergyBatch(
            energies=energies,
            stds=stds,
            shots=_shots_from_metadata(metadata) or shots,
            job_id=None if job_id_str is None else str(job_id_str),
            wall_time_s=elapsed,
        )

    def _run_pub(self, primitive: Any, params: FloatArray, shots: int | None) -> EnergyBatch:
        params = _as_2d(params)
        pub = (self._problem.circuit, self._problem.observable, params)
        precision = _precision(shots)
        start = time.perf_counter()
        job = (
            primitive.run([pub]) if precision is None else primitive.run([pub], precision=precision)
        )
        result = job.result()
        return self._unpack(job, result, params, time.perf_counter() - start, shots)

    def __enter__(self) -> Self:
        """No resources to acquire."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """No resources to release."""


class StatevectorEnergy(_PrimitiveEnergy):
    """Exact expectation values via :class:`qiskit.primitives.StatevectorEstimator`.

    Precision 0 means no sampling: results are deterministic up to floating point. The ISA
    problem may still be a transpiled circuit on a wider register (with the observable laid out
    accordingly); the statevector then has ``2**num_qubits`` amplitudes, which is fine up to
    ~25 qubits.
    """

    def __init__(self, problem: IsaProblem, *, seed: int | None = None) -> None:
        """Create the estimator."""
        from qiskit.primitives import StatevectorEstimator

        super().__init__(problem, {"kind": "statevector", "seed": seed}, seed=seed)
        self._primitive = StatevectorEstimator(default_precision=0.0, seed=seed)

    def evaluate(self, params: FloatArray, *, shots: int | None = None) -> EnergyBatch:
        """Exact energies; ``shots`` is ignored."""
        del shots
        return self._run_pub(self._primitive, params, None)


class AerEnergy(_PrimitiveEnergy):
    """Shot-sampled Aer simulation, ideal or with a device noise model.

    Energies are estimated exactly as on hardware: the circuit is measured in each qubit-wise
    commuting basis (three for the Heisenberg model), outcomes are sampled, and Pauli
    expectation values are averaged. Shot noise and readout error are therefore present.

    Args:
        problem: ISA problem (transpiled for ``noise_backend`` when one is given).
        shots: Shots per measurement basis.
        noise_backend: A fake ``BackendV2`` whose noise model and coupling map to import.
        noise_model: An explicit ``qiskit_aer.noise.NoiseModel``; overrides the backend's.
            Useful for isolating error sources, e.g. readout error only.
        seed: Seeds the per-call seed stream.
        method: Aer simulation method. ``"auto"`` picks ``statevector`` without noise (all
            shots are sampled from one final state), density matrices for noisy circuits up to
            :data:`DENSITY_MATRIX_MAX_QUBITS` active qubits, and Aer's own choice beyond.
    """

    def __init__(
        self,
        problem: IsaProblem,
        *,
        shots: int,
        noise_backend: Any | None = None,
        noise_model: Any | None = None,
        seed: int | None = None,
        method: AerMethod = "auto",
    ) -> None:
        """Create the estimator."""
        from qiskit.primitives import BackendEstimatorV2
        from qiskit_aer import AerSimulator

        if shots < 1:
            msg = f"shots must be >= 1, got {shots}"
            raise ValueError(msg)
        noisy = noise_backend is not None or noise_model is not None
        resolved: str = method
        if method == "auto":
            if not noisy:
                resolved = "statevector"
            elif _active_qubits(problem) <= DENSITY_MATRIX_MAX_QUBITS:
                resolved = "density_matrix"
            else:
                resolved = "automatic"
        sim_options: dict[str, Any] = {"method": resolved}
        if noise_model is not None:
            sim_options["noise_model"] = noise_model
        if noise_backend is not None:
            simulator = AerSimulator.from_backend(noise_backend, **sim_options)
        else:
            simulator = AerSimulator(**sim_options)
        self._primitive = BackendEstimatorV2(backend=simulator)
        kind = "aer_noisy" if noisy else "aer_ideal"
        super().__init__(
            problem,
            {
                "kind": kind,
                "noise_backend": None if noise_backend is None else str(noise_backend.name),
                "custom_noise_model": noise_model is not None,
                "method": resolved,
                "shots": shots,
                "seed": seed,
                "std": "primitive upper bound sum|c_i|sigma_i/sqrt(shots)",
            },
            shots=shots,
            seed=seed,
        )

    def evaluate(self, params: FloatArray, *, shots: int | None = None) -> EnergyBatch:
        """Sampled energies with a fresh simulator seed."""
        self._primitive.options.seed_simulator = self._next_seed()
        return self._run_pub(self._primitive, params, shots or self._configured_shots)


class LogicalNoiseEnergy:
    """Shot-sampled energies of a circuit carrying PRB 106 layer noise (``logical_noise``).

    The problem's circuit holds :class:`~kagome_vqe.noise.PauliChannel` instructions at the
    noise locations (see :func:`kagome_vqe.noise.layer_noise_circuit`). Each energy measures
    the observable's qubit-wise commuting groups (three for the Heisenberg model) with ``shots``
    shots each; every shot is an independent noise trajectory followed by one projective
    measurement, so sampling noise and gate noise are both present as on hardware.

    Engines: ``"trajectories"`` (:class:`~kagome_vqe.noise.PauliTrajectorySimulator`, only
    erroneous shots are simulated) or ``"aer"`` (the same circuit with Aer quantum errors,
    sampled by Aer through :class:`AerEnergy`).

    The reported ``std`` is the standard error of the estimate, ``sqrt(sum_g var_g / shots)``
    from the sampled outcomes (not the primitive's upper bound used by :class:`AerEnergy`).
    """

    def __init__(
        self,
        problem: IsaProblem,
        *,
        shots: int,
        engine: Literal["trajectories", "aer"] = "trajectories",
        seed: int | None = None,
        aer_method: AerMethod = "auto",
        noise: dict[str, Any] | None = None,
        fast_kernels: bool = False,
        readout_error: float = 0.0,
        readout: bool = False,
        symmetry: bool = False,
        readout_shots: int = 16384,
    ) -> None:
        """Create the estimator."""
        from kagome_vqe.noise import (
            PauliTrajectorySimulator,
            aer_circuit,
            measurement_bases,
            noise_locations,
        )

        if shots < 1:
            msg = f"shots must be >= 1, got {shots}"
            raise ValueError(msg)
        self._problem = problem
        self._shots = shots
        self._engine = engine
        self._seeds = np.random.default_rng(seed)
        self._aer: AerEnergy | None = None
        self._simulator: PauliTrajectorySimulator | None = None
        if engine == "aer":
            method: AerMethod = aer_method
            if method == "auto":
                method = (
                    "density_matrix"
                    if problem.circuit.num_qubits <= DENSITY_MATRIX_MAX_QUBITS
                    else "statevector"
                )
            self._aer = AerEnergy(
                problem.with_circuit(aer_circuit(problem.circuit)),
                shots=shots,
                seed=seed,
                method=method,
            )
        else:
            self._simulator = PauliTrajectorySimulator(problem.circuit, fast=fast_kernels)
            self._bases = measurement_bases(problem.observable)
        if (readout or symmetry or readout_error) and engine != "trajectories":
            msg = "readout error and counts-level mitigation need engine = 'trajectories'"
            raise ValueError(msg)
        self._readout_error = float(readout_error)
        self._readout = readout
        self._symmetry = symmetry
        self._readout_shots = readout_shots
        self._calibration: tuple[FloatArray, FloatArray] | None = None
        self._calibration_seed = None if seed is None else _calibration_seed(seed)
        self._description = {
            "kind": "logical_noise",
            "engine": engine,
            "fast_kernels": fast_kernels,
            "readout_error": self._readout_error,
            "readout_correction": readout,
            "symmetry_filter": symmetry,
            "noise": noise,
            "noise_locations": noise_locations(problem.circuit),
            "shots": shots,
            "seed": seed,
            "std": "standard error from sampled outcomes"
            if engine == "trajectories"
            else "primitive upper bound sum|c_i|sigma_i/sqrt(shots)",
        }

    @property
    def description(self) -> dict[str, Any]:
        """Backend and options."""
        if self._aer is not None:
            return {**self._description, "aer": self._aer.description}
        out = dict(self._description)
        if self._calibration is not None:
            e0, e1 = self._calibration
            out["readout_calibration"] = {
                "shots_per_circuit": self._readout_shots,
                "mean_e0": float(np.mean(e0)),
                "mean_e1": float(np.mean(e1)),
            }
        return out

    def _calibrate(self) -> FloatArray:
        """Per-qubit inverse confusion matrices from all-0 / all-1 preparations (once)."""
        from kagome_vqe.mitigation.counts import ReadoutCalibration

        if self._calibration is None:
            n = self._problem.circuit.num_qubits
            rng = np.random.default_rng(self._calibration_seed)
            q, shots = self._readout_error, self._readout_shots
            e0 = rng.binomial(shots, q, size=n) / shots
            e1 = rng.binomial(shots, q, size=n) / shots
            self._calibration = (e0, e1)
        e0, e1 = self._calibration
        return ReadoutCalibration(e0, e1).inverse_matrices()

    def _basis_statistics(
        self, counts: NDArray[np.int64], diagonal: FloatArray, inverses: FloatArray | None
    ) -> tuple[float, float]:
        """Mean and variance of one basis's estimate (readout correction and symmetry filter).

        Correct-then-filter as in :mod:`kagome_vqe.mitigation.counts`: the estimate is
        ``sum_mask (A^-1 p) d / sum_mask (A^-1 p)``. Both sums are per-shot means of
        ``u = A^-T (mask d)`` and ``v = A^-T mask``, so the ratio's variance follows from the
        per-shot covariance (delta method); without correction or filter it reduces to the plain
        mean and variance of ``d``.
        """
        from kagome_vqe.mitigation.counts import _apply_inverse

        shots = float(counts.sum())
        dim = counts.size
        mask = np.ones(dim)
        if self._symmetry:
            n = self._problem.circuit.num_qubits
            weight = np.bitwise_count(np.arange(dim, dtype=np.int64)).astype(np.int64)
            mask = (weight == n // 2).astype(float)
        u, v = diagonal * mask, mask
        if inverses is not None:
            transposed = np.transpose(inverses, (0, 2, 1))
            u, v = _apply_inverse(u, transposed), _apply_inverse(v, transposed)
        p = counts / shots
        u_mean, v_mean = float(p @ u), float(p @ v)
        if not self._symmetry and inverses is None:
            return u_mean, max(float(p @ u**2) - u_mean**2, 0.0) / shots
        if v_mean <= 0:
            log.warning("symmetry filter kept no weight; using the unfiltered estimate")
            return float(p @ diagonal), float(p @ diagonal**2 - (p @ diagonal) ** 2) / shots
        ratio = u_mean / v_mean
        var_u = float(p @ u**2) - u_mean**2
        var_v = float(p @ v**2) - v_mean**2
        cov = float(p @ (u * v)) - u_mean * v_mean
        variance = (var_u - 2 * ratio * cov + ratio**2 * var_v) / (shots * v_mean**2)
        return ratio, max(variance, 0.0)

    def _readout_noise(
        self, counts: NDArray[np.int64], rng: np.random.Generator
    ) -> NDArray[np.int64]:
        """Flip every measured bit independently with probability ``readout_error``."""
        n = self._problem.circuit.num_qubits
        outcomes = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
        flips = rng.random((outcomes.size, n)) < self._readout_error
        outcomes ^= (flips.astype(np.int64) << np.arange(n, dtype=np.int64)).sum(axis=1)
        return np.bincount(outcomes, minlength=counts.size).astype(np.int64)

    def evaluate(self, params: FloatArray, *, shots: int | None = None) -> EnergyBatch:
        """Sampled energies, every row with fresh noise and measurement randomness."""
        n_shots = shots or self._shots
        if self._aer is not None:
            return self._aer.evaluate(params, shots=n_shots)
        assert self._simulator is not None
        rows = _as_2d(params)
        start = time.perf_counter()
        rng = np.random.default_rng(int(self._seeds.integers(0, 2**31 - 1)))
        energies = np.empty(rows.shape[0])
        stds = np.empty(rows.shape[0])
        letters = [basis for basis, _ in self._bases]
        inverses = self._calibrate() if self._readout else None
        for i, theta in enumerate(rows):
            histograms = self._simulator.sample_measurements(theta, letters, n_shots, rng)
            mean = 0.0
            variance = 0.0
            for counts, (_, diagonal) in zip(histograms, self._bases, strict=True):
                if self._readout_error > 0:
                    counts = self._readout_noise(counts, rng)  # noqa: PLW2901
                first, var = self._basis_statistics(counts, diagonal, inverses)
                mean += first
                variance += var
            energies[i] = mean
            stds[i] = math.sqrt(variance)
        return EnergyBatch(
            energies=energies,
            stds=stds,
            shots=n_shots,
            job_id=None,
            wall_time_s=time.perf_counter() - start,
        )

    def __enter__(self) -> Self:
        """No resources to acquire."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """No resources to release."""


class RuntimeSession:
    """Lifecycle, seeding and retries for one qiskit-ibm-runtime primitive.

    Owns the execution mode (job / session / batch) and an ``EstimatorV2`` or ``SamplerV2``
    bound to it. :meth:`run` submits one job and retries transient failures: a job that fails or
    exceeds ``job_timeout_s`` is cancelled and resubmitted up to ``retries`` times with
    exponential back-off; if the session was closed meanwhile a new one is opened (which
    re-enters the queue, so this is logged loudly). In local (fake) mode every job gets a fresh
    simulator seed from ``seeds``.

    Args:
        backend: Real or fake ``BackendV2``.
        cfg: Backend configuration (shots, resilience level, mode, retries, ...).
        primitive: ``"estimator"`` or ``"sampler"``.
        seeds: Seed stream for local mode.
    """

    def __init__(
        self,
        backend: Any,
        cfg: BackendConfig,
        primitive: Literal["estimator", "sampler"],
        seeds: np.random.Generator,
    ) -> None:
        """Create the session wrapper; nothing is opened until ``__enter__``."""
        self._backend = backend
        self._cfg = cfg
        self._kind = primitive
        self._seeds = seeds
        self._mode: Any = None
        self.primitive: Any = None

    def _next_seed(self) -> int:
        return int(self._seeds.integers(0, 2**31 - 1))

    def _options(self) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if self._kind == "estimator":
            options["resilience_level"] = self._cfg.resilience_level
        if self._cfg.shots is not None:
            options["default_shots"] = self._cfg.shots
        if self._cfg.kind == "fake":
            options["simulator"] = {"seed_simulator": self._next_seed()}
        return options

    def _open_mode(self) -> Any:
        from qiskit_ibm_runtime import Batch, Session

        if self._cfg.mode == "session":
            return Session(backend=self._backend, max_time=self._cfg.max_time)
        if self._cfg.mode == "batch":
            return Batch(backend=self._backend, max_time=self._cfg.max_time)
        return self._backend

    def _make_primitive(self) -> Any:
        options = self._options()
        if self._kind == "sampler":
            from qiskit_ibm_runtime import SamplerV2

            return SamplerV2(mode=self._mode, options=options)
        if self._cfg.estimator_impl == "executor":
            from qiskit_ibm_runtime.executor_estimator import Estimator

            return Estimator(mode=self._mode, options=options)
        from qiskit_ibm_runtime import EstimatorV2

        return EstimatorV2(mode=self._mode, options=options)

    def __enter__(self) -> Self:
        """Open the session/batch and create the primitive."""
        self._mode = self._open_mode()
        self.primitive = self._make_primitive()
        log.info(
            "runtime %s ready on %s (mode=%s, shots=%s)",
            self._kind,
            self._backend.name,
            self._cfg.mode,
            self._cfg.shots,
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the session/batch."""
        close = getattr(self._mode, "close", None)
        if callable(close):
            close()
        self._mode = None
        self.primitive = None

    def _mode_is_open(self) -> bool:
        status = getattr(self._mode, "status", None)
        if not callable(status):
            return True  # plain backend mode has no lifecycle
        try:
            value = status()
        except Exception:
            return True
        return value is None or str(value).lower() != "closed"  # local mode reports None

    def _reopen(self) -> None:
        log.warning(
            "runtime %s closed; opening a new one (this re-enters the queue)", self._cfg.mode
        )
        close = getattr(self._mode, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()
        self._mode = self._open_mode()
        self.primitive = self._make_primitive()

    def _refresh_seed(self) -> None:
        """Local (fake) mode only: new simulator seed for every job."""
        if self._cfg.kind != "fake":
            return
        simulator = getattr(getattr(self.primitive, "options", None), "simulator", None)
        if simulator is None:
            return
        with contextlib.suppress(AttributeError, TypeError, ValueError):
            simulator.seed_simulator = self._next_seed()

    def run(self, pubs: list[Any], **kwargs: Any) -> tuple[Any, Any, float]:
        """Submit one job with retries; returns ``(job, result, seconds)``."""
        if self.primitive is None:
            msg = "the runtime session must be used as a context manager"
            raise RuntimeError(msg)
        last_error: BaseException | None = None
        for attempt in range(self._cfg.retries + 1):
            job = None
            start = time.perf_counter()
            try:
                if not self._mode_is_open():
                    self._reopen()
                self._refresh_seed()
                job = self.primitive.run(pubs, **kwargs)
                result = _job_result(job, self._cfg.job_timeout_s)
                return job, result, time.perf_counter() - start
            except (KeyboardInterrupt, SystemExit):
                self._cancel(job)
                raise
            except _transient_errors() as exc:
                last_error = exc
                self._cancel(job)
                if attempt >= self._cfg.retries:
                    break
                delay = min(60.0, 2.0**attempt)
                log.warning(
                    "runtime job failed (attempt %d/%d): %s; retrying in %.0fs",
                    attempt + 1,
                    self._cfg.retries + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)
        msg = f"runtime job failed after {self._cfg.retries + 1} attempts"
        raise RuntimeError(msg) from last_error

    @staticmethod
    def _cancel(job: Any) -> None:
        if job is None:
            return
        try:
            status = job.status()
            if str(status).upper() not in {"DONE", "ERROR", "CANCELLED"}:
                job.cancel()
        except Exception:
            log.debug("could not cancel job", exc_info=True)


class RuntimeEnergy(_PrimitiveEnergy):
    """qiskit-ibm-runtime estimator on a real device or a fake backend in local mode.

    Execution modes, error-mitigation options (``resilience_level``) and retries are handled by
    :class:`RuntimeSession`.

    Args:
        problem: ISA problem transpiled for ``backend``.
        backend: Real or fake ``BackendV2``.
        cfg: Backend configuration (shots, resilience level, mode, retries, ...).
        seed: Seeds the per-call simulator seed stream in local (fake) mode; ignored on
            hardware.
    """

    def __init__(
        self, problem: IsaProblem, backend: Any, cfg: BackendConfig, *, seed: int | None = None
    ) -> None:
        """Create the estimator; the session is opened in ``__enter__``."""
        super().__init__(
            problem,
            {
                "kind": cfg.kind,
                "backend": str(backend.name),
                "mode": cfg.mode,
                "shots": cfg.shots,
                "resilience_level": cfg.resilience_level,
                "estimator_impl": cfg.estimator_impl,
                "max_time": cfg.max_time,
            },
            shots=cfg.shots,
            seed=seed,
        )
        self._cfg = cfg
        self._session = RuntimeSession(backend, cfg, "estimator", self._seeds)

    def __enter__(self) -> Self:
        """Open the session/batch."""
        self._session.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the session/batch."""
        self._session.__exit__(exc_type, exc, tb)

    def evaluate(self, params: FloatArray, *, shots: int | None = None) -> EnergyBatch:
        """Submit one PUB with retries."""
        if self._session.primitive is None:
            msg = "RuntimeEnergy must be used as a context manager"
            raise RuntimeError(msg)
        params = _as_2d(params)
        precision = _precision(shots)
        pub = (self._problem.circuit, self._problem.observable, params)
        kwargs = {} if precision is None else {"precision": precision}
        job, result, elapsed = self._session.run([pub], **kwargs)
        return self._unpack(job, result, params, elapsed, shots or self._cfg.shots)


def aer_noise_model(backend: Any, noise: str) -> Any | None:
    """Noise model restricted to one error source; ``None`` means the backend's full model."""
    if noise == "full":
        return None
    from qiskit_aer.noise import NoiseModel

    gates = noise == "gates"
    return NoiseModel.from_backend(
        backend, gate_error=gates, thermal_relaxation=gates, readout_error=not gates
    )


def _base_estimator(
    cfg: BackendConfig,
    problem: IsaProblem,
    backend: Any | None,
    seed: int | None,
    counts: MitigationConfig | None = None,
) -> EnergyEstimator:
    if cfg.kind == "statevector":
        return StatevectorEnergy(problem, seed=seed)
    assert cfg.shots is not None  # guaranteed by BackendConfig validation
    if cfg.kind == "aer_ideal":
        return AerEnergy(problem, shots=cfg.shots, seed=seed, method=cfg.aer_method)
    if cfg.kind == "logical_noise":
        assert cfg.logical_noise is not None  # guaranteed by BackendConfig validation
        return LogicalNoiseEnergy(
            problem,
            shots=cfg.shots,
            engine=cfg.logical_noise.engine,
            seed=seed,
            aer_method=cfg.aer_method,
            noise=cfg.logical_noise.model_dump(),
            fast_kernels=cfg.logical_noise.fast_kernels,
            readout_error=cfg.logical_noise.readout_error,
            readout=counts.readout if counts is not None else False,
            symmetry=counts.symmetry if counts is not None else False,
            readout_shots=counts.readout_shots if counts is not None else 16384,
        )
    if cfg.kind == "aer_noisy":
        return AerEnergy(
            problem,
            shots=cfg.shots,
            noise_backend=backend,
            noise_model=aer_noise_model(backend, cfg.noise),
            seed=seed,
            method=cfg.aer_method,
        )
    if backend is None:
        msg = f"backend.kind={cfg.kind!r} needs a backend object"
        raise ValueError(msg)
    return RuntimeEnergy(problem, backend, cfg, seed=seed)


def _calibration_seed(seed: int) -> int:
    """Seed of a logical-noise estimator's one-off readout calibration."""
    return int(np.random.SeedSequence([seed, 7_000_003]).generate_state(1)[0])


def _spawn_seed(seed: int, index: int) -> int:
    """A seed for the ``index``-th sub-estimator, independent of the others."""
    return int(np.random.SeedSequence([seed, index]).generate_state(1)[0])


def make_estimator(
    cfg: BackendConfig,
    problem: IsaProblem,
    backend: Any | None,
    *,
    seed: int | None = None,
    settings: Settings | None = None,
    mitigation: MitigationConfig | None = None,
    twin: tuple[FloatArray, float] | None = None,
) -> EnergyEstimator:
    """Pick the estimator for ``cfg.kind`` and wrap it in the requested mitigation.

    The stack is built innermost first: counts-level processing (readout correction, symmetry
    filtering) replaces the plain estimator when requested; ZNE wraps that; twin rescaling
    wraps everything.

    Args:
        cfg: Backend configuration.
        problem: ISA problem already compiled for ``backend``.
        backend: The object from :func:`kagome_vqe.backends.resolve_backend` (``None`` for
            statevector).
        seed: Seeds the estimator's per-call seed stream.
        settings: IBM credentials (unused here; the backend object already carries them).
        mitigation: Error-mitigation settings; ``None`` or all-off gives the plain estimator.
        twin: Circuit parameters and noise-free energy of the rescaling twin; required when
            ``mitigation.rescale`` is set.
    """
    del settings  # reserved for future per-job options
    if mitigation is None or not mitigation.enabled:
        return _base_estimator(cfg, problem, backend, seed)

    from kagome_vqe.mitigation import CountsEnergy, TwinRescaledEnergy, ZneEnergy

    def inner(prob: IsaProblem) -> EnergyEstimator:
        if cfg.kind == "logical_noise":  # counts-level mitigation inside the sampler
            return _base_estimator(cfg, prob, backend, seed, mitigation)
        if mitigation.needs_counts:
            return CountsEnergy(
                prob,
                cfg,
                backend,
                readout=mitigation.readout,
                readout_shots=mitigation.readout_shots,
                symmetry=mitigation.symmetry,
                seed=seed,
            )
        return _base_estimator(cfg, prob, backend, seed)

    estimator: EnergyEstimator
    zne = mitigation.zne
    if zne is not None and cfg.kind == "logical_noise":
        from kagome_vqe.noise import fold_noise_layers, scale_noise_probability

        def scaled(circuit: QuantumCircuit, factor: float) -> QuantumCircuit:
            if zne.noise_scaling == "probability":
                return scale_noise_probability(circuit, factor)
            return fold_noise_layers(circuit, (round(factor) - 1) // 2)

        def inner_at(prob: IsaProblem, index: int) -> EnergyEstimator:
            # Independent randomness per noise scale, as separate hardware jobs would have.
            scale_seed = None if seed is None else _spawn_seed(seed, index)
            return _base_estimator(cfg, prob, backend, scale_seed, mitigation)

        estimator = ZneEnergy(
            problem,
            inner,
            zne.scales,
            zne.extrapolation,
            scale_circuit=scaled,
            make_inner_at=inner_at,
            noise_scaling=zne.noise_scaling,
        )
    elif zne is not None:
        estimator = ZneEnergy(problem, inner, zne.scales, zne.extrapolation)
    else:
        estimator = inner(problem)
    if mitigation.rescale is not None:
        if twin is None:
            msg = "mitigation.rescale needs twin parameters and their noise-free energy"
            raise ValueError(msg)
        twin_params, twin_energy = twin
        estimator = TwinRescaledEnergy(
            estimator,
            twin_params,
            twin_energy,
            twin_shots=mitigation.rescale.shots or cfg.final_shots or cfg.shots,
        )
    return estimator
