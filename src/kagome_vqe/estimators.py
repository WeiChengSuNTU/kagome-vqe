"""Energy evaluation on simulators, fake backends and IBM Quantum.

Every backend family is wrapped in an :class:`EnergyEstimator` that takes a batch of parameter
vectors and returns a batch of energies. All implementations submit the batch as **one**
primitive unified bloc (PUB) with broadcast parameters, so an optimiser iteration is one job,
whatever the backend.

Estimators are context managers because the runtime ones own a ``Session``/``Batch``::

    with make_estimator(cfg.backend, problem, seed=cfg.seed) as estimator:
        energies = estimator.evaluate(params).energies
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import time
from dataclasses import dataclass
from functools import cache
from types import TracebackType
from typing import Any, Protocol, Self, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from kagome_vqe.config import BackendConfig, Settings
from kagome_vqe.transpile import IsaProblem

__all__ = [
    "AerEnergy",
    "EnergyBatch",
    "EnergyEstimator",
    "RuntimeEnergy",
    "StatevectorEnergy",
    "make_estimator",
]

log = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class EnergyBatch:
    """Energies for a batch of parameter vectors.

    Attributes:
        energies: Shape ``(k,)``.
        stds: Standard errors, shape ``(k,)``; zeros for exact evaluation.
        shots: Shots per expectation value as reported by the primitive, if known.
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

    def evaluate(self, params: FloatArray) -> EnergyBatch:
        """``params`` has shape ``(k, n)``; returns ``k`` energies."""
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


class _PrimitiveEnergy:
    """Shared logic: run one PUB on a V2 estimator and unpack the result."""

    def __init__(
        self, problem: IsaProblem, description: dict[str, Any], *, shots: int | None = None
    ) -> None:
        self._problem = problem
        self._description = description
        self._configured_shots = shots

    @property
    def description(self) -> dict[str, Any]:
        """Backend and options."""
        return dict(self._description)

    def _run_pub(self, primitive: Any, params: FloatArray) -> EnergyBatch:
        params = _as_2d(params)
        pub = (self._problem.circuit, self._problem.observable, params)
        start = time.perf_counter()
        job = primitive.run([pub])
        result = job.result()
        elapsed = time.perf_counter() - start
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
            shots=_shots_from_metadata(metadata) or self._configured_shots,
            job_id=None if job_id_str is None else str(job_id_str),
            wall_time_s=elapsed,
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

        super().__init__(problem, {"kind": "statevector", "seed": seed})
        self._primitive = StatevectorEstimator(default_precision=0.0, seed=seed)

    def evaluate(self, params: FloatArray) -> EnergyBatch:
        """Exact energies."""
        return self._run_pub(self._primitive, params)


class AerEnergy(_PrimitiveEnergy):
    """Shot-based Aer simulation, ideal or with a fake backend's noise model.

    Args:
        problem: ISA problem (transpiled for the fake backend when ``noise_backend`` is given).
        shots: Shots per expectation value; ``None`` uses Aer's default precision.
        noise_backend: A fake ``BackendV2`` whose noise model and coupling map to import.
        seed: Simulator seed.
    """

    def __init__(
        self,
        problem: IsaProblem,
        *,
        shots: int | None,
        noise_backend: Any | None = None,
        seed: int | None = None,
    ) -> None:
        """Create the estimator."""
        from qiskit_aer.primitives import EstimatorV2

        run_options: dict[str, Any] = {}
        if shots is not None:
            run_options["shots"] = shots
        if seed is not None:
            run_options["seed_simulator"] = seed
        if noise_backend is None:
            kind = "aer_ideal"
            self._primitive = EstimatorV2(options={"run_options": run_options})
        else:
            kind = "aer_noisy"
            self._primitive = EstimatorV2.from_backend(
                noise_backend, options={"run_options": run_options}
            )
        super().__init__(
            problem,
            {
                "kind": kind,
                "noise_backend": None if noise_backend is None else str(noise_backend.name),
                "shots": shots,
                "seed": seed,
            },
            shots=shots,
        )

    def evaluate(self, params: FloatArray) -> EnergyBatch:
        """Sampled energies."""
        return self._run_pub(self._primitive, params)


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
        seed: ``simulator.seed_simulator`` for local/fake execution; ignored on hardware.
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
        )
        self._backend = backend
        self._cfg = cfg
        self._seed = seed
        self._mode: Any = None
        self._primitive: Any = None

    # -- lifecycle ----------------------------------------------------------------------------

    def _options(self) -> dict[str, Any]:
        options: dict[str, Any] = {"resilience_level": self._cfg.resilience_level}
        if self._cfg.shots is not None:
            options["default_shots"] = self._cfg.shots
        if self._seed is not None and self._cfg.kind == "fake":
            options["simulator"] = {"seed_simulator": self._seed}
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

    # -- evaluation ---------------------------------------------------------------------------

    def evaluate(self, params: FloatArray) -> EnergyBatch:
        """Submit one PUB with retries."""
        if self._primitive is None:
            msg = "RuntimeEnergy must be used as a context manager"
            raise RuntimeError(msg)
        params = _as_2d(params)
        last_error: BaseException | None = None
        for attempt in range(self._cfg.retries + 1):
            job = None
            start = time.perf_counter()
            try:
                if not self._mode_is_open():
                    self._reopen()
                pub = (self._problem.circuit, self._problem.observable, params)
                job = self._primitive.run([pub])
                result = _job_result(job, self._cfg.job_timeout_s)
                return self._unpack(job, result, params, time.perf_counter() - start)
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

    def _unpack(self, job: Any, result: Any, params: FloatArray, elapsed: float) -> EnergyBatch:
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
            shots=_shots_from_metadata(metadata) or self._cfg.shots,
            job_id=None if job_id_str is None else str(job_id_str),
            wall_time_s=elapsed,
        )


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
        seed: Simulator seed.
        settings: IBM credentials (unused here; the backend object already carries them).
    """
    del settings  # reserved for future per-job options
    if cfg.kind == "statevector":
        return StatevectorEnergy(problem, seed=seed)
    if cfg.kind == "aer_ideal":
        return AerEnergy(problem, shots=cfg.shots, seed=seed)
    if cfg.kind == "aer_noisy":
        return AerEnergy(problem, shots=cfg.shots, noise_backend=backend, seed=seed)
    if backend is None:
        msg = f"backend.kind={cfg.kind!r} needs a backend object"
        raise ValueError(msg)
    return RuntimeEnergy(problem, backend, cfg, seed=seed)
