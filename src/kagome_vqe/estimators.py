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
from typing import Any, Literal, Protocol, Self, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from kagome_vqe.config import BackendConfig, Settings
from kagome_vqe.transpile import IsaProblem

__all__ = [
    "DENSITY_MATRIX_MAX_QUBITS",
    "AerEnergy",
    "EnergyBatch",
    "EnergyEstimator",
    "RuntimeEnergy",
    "StatevectorEnergy",
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
    """

    energies: FloatArray
    stds: FloatArray
    shots: int | None = None
    job_id: str | None = None
    wall_time_s: float = 0.0


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


class RuntimeEnergy(_PrimitiveEnergy):
    """qiskit-ibm-runtime estimator on a real device or a fake backend in local mode.

    Handles execution modes (job / session / batch), error-mitigation options, and retries:
    a job that fails or exceeds ``job_timeout_s`` is cancelled and resubmitted up to
    ``retries`` times with exponential back-off; if the session was closed meanwhile a new one
    is opened (which re-enters the queue, so this is logged loudly).

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
        self._backend = backend
        self._cfg = cfg
        self._mode: Any = None
        self._primitive: Any = None

    # -- lifecycle ----------------------------------------------------------------------------

    def _options(self) -> dict[str, Any]:
        options: dict[str, Any] = {"resilience_level": self._cfg.resilience_level}
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
        if self._cfg.estimator_impl == "executor":
            from qiskit_ibm_runtime.executor_estimator import Estimator

            return Estimator(mode=self._mode, options=options)
        from qiskit_ibm_runtime import EstimatorV2

        return EstimatorV2(mode=self._mode, options=options)

    def __enter__(self) -> Self:
        """Open the session/batch and create the primitive."""
        self._mode = self._open_mode()
        self._primitive = self._make_primitive()
        log.info(
            "runtime estimator ready on %s (mode=%s, shots=%s, resilience=%d)",
            self._backend.name,
            self._cfg.mode,
            self._cfg.shots,
            self._cfg.resilience_level,
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
        self._primitive = None

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
        self._primitive = self._make_primitive()

    def _refresh_seed(self) -> None:
        """Local (fake) mode only: new simulator seed for every job."""
        if self._cfg.kind != "fake":
            return
        simulator = getattr(getattr(self._primitive, "options", None), "simulator", None)
        if simulator is None:
            return
        with contextlib.suppress(AttributeError, TypeError, ValueError):
            simulator.seed_simulator = self._next_seed()

    # -- evaluation ---------------------------------------------------------------------------

    def evaluate(self, params: FloatArray, *, shots: int | None = None) -> EnergyBatch:
        """Submit one PUB with retries."""
        if self._primitive is None:
            msg = "RuntimeEnergy must be used as a context manager"
            raise RuntimeError(msg)
        params = _as_2d(params)
        precision = _precision(shots)
        last_error: BaseException | None = None
        for attempt in range(self._cfg.retries + 1):
            job = None
            start = time.perf_counter()
            try:
                if not self._mode_is_open():
                    self._reopen()
                self._refresh_seed()
                pub = (self._problem.circuit, self._problem.observable, params)
                job = (
                    self._primitive.run([pub])
                    if precision is None
                    else self._primitive.run([pub], precision=precision)
                )
                result = _job_result(job, self._cfg.job_timeout_s)
                return self._unpack(
                    job, result, params, time.perf_counter() - start, shots or self._cfg.shots
                )
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
        msg = f"runtime evaluation failed after {self._cfg.retries + 1} attempts"
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


def make_estimator(
    cfg: BackendConfig,
    problem: IsaProblem,
    backend: Any | None,
    *,
    seed: int | None = None,
    settings: Settings | None = None,
) -> EnergyEstimator:
    """Pick the estimator implementation for ``cfg.kind``.

    Args:
        cfg: Backend configuration.
        problem: ISA problem already compiled for ``backend``.
        backend: The object from :func:`kagome_vqe.backends.resolve_backend` (``None`` for
            statevector).
        seed: Seeds the estimator's per-call seed stream.
        settings: IBM credentials (unused here; the backend object already carries them).
    """
    del settings  # reserved for future per-job options
    if cfg.kind == "statevector":
        return StatevectorEnergy(problem, seed=seed)
    assert cfg.shots is not None  # guaranteed by BackendConfig validation
    if cfg.kind == "aer_ideal":
        return AerEnergy(problem, shots=cfg.shots, seed=seed, method=cfg.aer_method)
    if cfg.kind == "aer_noisy":
        return AerEnergy(
            problem, shots=cfg.shots, noise_backend=backend, seed=seed, method=cfg.aer_method
        )
    if backend is None:
        msg = f"backend.kind={cfg.kind!r} needs a backend object"
        raise ValueError(msg)
    return RuntimeEnergy(problem, backend, cfg, seed=seed)
