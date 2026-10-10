"""The VQE loop: wire lattice, ansatz, estimator and optimiser together and record a trace.

:func:`prepare` does everything that needs no quantum execution (build objects, transpile,
compute the exact reference energy) and is what ``kagome-vqe run --dry-run`` reports.
:func:`run_vqe` then optimises and returns a :class:`VQERunResult` that
:mod:`kagome_vqe.results` knows how to persist.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qiskit.quantum_info import SparsePauliOp

from kagome_vqe.ansatz import Ansatz
from kagome_vqe.ansatz.unshare import ParameterMap, unshare_parameters
from kagome_vqe.backends import resolve_backend
from kagome_vqe.config import RunConfig, Settings
from kagome_vqe.estimators import EnergyEstimator, StatevectorEnergy, make_estimator
from kagome_vqe.hamiltonian import exact_spectrum, heisenberg_hamiltonian, relative_error
from kagome_vqe.lattice import Lattice
from kagome_vqe.optimizers import IterationInfo, Optimizer
from kagome_vqe.transpile import IsaProblem, compile_for_backend

__all__ = [
    "IDEAL_TRACKING_MAX_QUBITS",
    "EnergyObjective",
    "FixedPointResult",
    "IdealPoint",
    "Prepared",
    "TraceRow",
    "VQERunResult",
    "evaluate_fixed",
    "prepare",
    "run_vqe",
]

IDEAL_TRACKING_MAX_QUBITS = 20
"""Above this many qubits the per-iteration statevector check is skipped (memory and time)."""

log = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TraceRow:
    """One energy evaluation.

    Attributes:
        eval_idx: Running index over all evaluations (0-based).
        iteration: Optimiser iteration the evaluation belongs to (1-based; 0 = pre-loop).
        energy: Measured energy.
        std: Standard error reported by the primitive (0 for exact).
        shots: Shots used, if known.
        wall_time_s: Time of the primitive call that produced this value, divided evenly over
            the evaluations of the batch.
        job_id: Runtime job id, if any.
    """

    eval_idx: int
    iteration: int
    energy: float
    std: float
    shots: int | None
    wall_time_s: float
    job_id: str | None


class EnergyObjective:
    """An :class:`~kagome_vqe.optimizers.ExpandableObjective` that records every evaluation.

    Optimisers call :meth:`batch` with ansatz parameters ``theta``; parameter-shift gradients
    call :meth:`batch_expanded` with circuit parameters ``phi`` (one per gate occurrence). Both
    forward one batch to the estimator and append one :class:`TraceRow` per row, tagged with
    the current iteration (which :meth:`on_iteration` advances).

    Args:
        estimator: Evaluates circuit-parameter rows.
        record_params: Keep the ansatz parameters of every evaluation. For parameter-shift rows
            the point the gradient was taken at is recorded.
        parameter_map: Map from ansatz to circuit parameters; identity if omitted.
    """

    def __init__(
        self,
        estimator: EnergyEstimator,
        *,
        record_params: bool = True,
        parameter_map: ParameterMap | None = None,
    ) -> None:
        """Wrap ``estimator``."""
        self._estimator = estimator
        self._record_params = record_params
        self._map = parameter_map
        self.trace: list[TraceRow] = []
        self.params: list[FloatArray] = []
        self._iteration = 1
        self._listeners: list[Callable[[TraceRow], None]] = []

    def add_listener(self, fn: Callable[[TraceRow], None]) -> None:
        """Call ``fn`` after every evaluation (progress bars, live logging)."""
        self._listeners.append(fn)

    @property
    def nfev(self) -> int:
        """Evaluations so far."""
        return len(self.trace)

    @property
    def parameter_map(self) -> ParameterMap:
        """Map from ansatz parameters to the estimator's circuit parameters."""
        if self._map is None:
            msg = "parameter map unknown until the first batch; pass parameter_map explicitly"
            raise RuntimeError(msg)
        return self._map

    @property
    def shots_spent(self) -> int:
        """Shots per measurement basis summed over all evaluations so far."""
        return sum(row.shots or 0 for row in self.trace)

    @property
    def best(self) -> TraceRow | None:
        """Lowest energy seen so far."""
        return min(self.trace, key=lambda r: r.energy, default=None)

    def on_iteration(self, info: IterationInfo) -> None:
        """Optimiser callback: subsequent evaluations belong to the next iteration."""
        self._iteration = info.nit + 1

    def batch(self, xs: FloatArray) -> FloatArray:
        """Evaluate a ``(k, n)`` batch of ansatz parameters and record it."""
        xs = np.atleast_2d(np.asarray(xs, dtype=float))
        if self._map is None:
            self._map = ParameterMap.identity(xs.shape[1])
        return self._evaluate(self._map.expand(xs), xs)

    def batch_expanded(self, phis: FloatArray, *, base: FloatArray | None = None) -> FloatArray:
        """Evaluate a ``(k, K)`` batch of circuit parameters and record it."""
        phis = np.atleast_2d(np.asarray(phis, dtype=float))
        thetas = phis if base is None else np.repeat(np.asarray(base, float)[None, :], len(phis), 0)
        return self._evaluate(phis, thetas)

    def _evaluate(self, phis: FloatArray, thetas: FloatArray) -> FloatArray:
        result = self._estimator.evaluate(phis)
        per_eval = result.wall_time_s / max(1, phis.shape[0])
        for x, energy, std in zip(thetas, result.energies, result.stds, strict=True):
            row = TraceRow(
                eval_idx=len(self.trace),
                iteration=self._iteration,
                energy=float(energy),
                std=float(std),
                shots=result.shots,
                wall_time_s=per_eval,
                job_id=result.job_id,
            )
            self.trace.append(row)
            if self._record_params:
                self.params.append(np.array(x, dtype=float))
            for fn in self._listeners:
                fn(row)
        return np.asarray(result.energies, dtype=float)

    def __call__(self, x: FloatArray) -> float:
        """Evaluate one point."""
        return float(self.batch(np.asarray(x, dtype=float)[None, :])[0])


@dataclass(slots=True)
class Prepared:
    """Everything about a run that does not require executing circuits.

    Attributes:
        config: The run configuration.
        lattice: Spin graph.
        hamiltonian: Logical-qubit Hamiltonian.
        ansatz: Ansatz factory.
        problem: ISA circuit and observable for the backend.
        backend: Backend object (``None`` for statevector).
        exact_spectrum: Lowest three exact eigenvalues (reference).
        projected_evaluations: Objective evaluations the optimiser will use, if predictable.
        parameter_map: Map from ansatz parameters to the compiled circuit's parameters. The
            identity unless parameter shift required one parameter per gate occurrence.
        twin: Twin for rescaling mitigation as ``(ansatz parameters, noise-free energy)``, or
            ``None`` when rescaling is off.
    """

    config: RunConfig
    lattice: Lattice
    hamiltonian: SparsePauliOp
    ansatz: Ansatz
    problem: IsaProblem
    backend: Any | None
    exact_spectrum: FloatArray
    projected_evaluations: int | None
    parameter_map: ParameterMap = field(default_factory=lambda: ParameterMap.identity(0))
    twin: tuple[FloatArray, float] | None = None

    @property
    def circuit_twin(self) -> tuple[FloatArray, float] | None:
        """The twin with its parameters expanded to circuit-parameter space."""
        if self.twin is None:
            return None
        return self.parameter_map.expand(self.twin[0]), self.twin[1]

    @property
    def exact_energy(self) -> float:
        """Exact ground-state energy."""
        return float(self.exact_spectrum[0])

    def summary(self) -> dict[str, Any]:
        """Dry-run report."""
        return {
            "name": self.config.name,
            "lattice": {
                "name": self.lattice.name,
                "num_sites": self.lattice.num_sites,
                "num_bonds": self.lattice.num_bonds,
                "site_order": self.lattice.site_order,
                "ring_bond_fraction": self.lattice.ring_bond_fraction(),
            },
            "hamiltonian_terms": len(self.hamiltonian),
            "exact_spectrum": [float(v) for v in self.exact_spectrum],
            "ansatz": {
                "name": self.ansatz.name,
                "num_parameters": self.ansatz.num_parameters,
                "circuit_parameters": self.parameter_map.num_occurrences,
            },
            "optimizer": self.config.optimizer.model_dump(),
            "backend": self.config.backend.label(),
            "mitigation": self.config.mitigation.label(),
            "twin": None
            if self.twin is None
            else {"params": self.twin[0].tolist(), "ideal_energy": self.twin[1]},
            "isa": self.problem.stats(),
            "projected_evaluations": self.projected_evaluations,
        }


@dataclass(frozen=True, slots=True)
class IdealPoint:
    """Noise-free energy of the optimiser's parameters after an iteration.

    Attributes:
        iteration: Optimiser iteration (0 = initial point).
        nfev: Energy evaluations spent so far.
        shots: Circuit shots spent so far, over all measurement bases.
        ideal_energy: Statevector energy of the parameters, independent of noise and shots.
    """

    iteration: int
    nfev: int
    shots: int
    ideal_energy: float


@dataclass(slots=True)
class VQERunResult:
    """Outcome of :func:`run_vqe`.

    Attributes:
        config: The configuration that produced it.
        energy: The optimiser's own last evaluation at its final parameters (regular shots).
        best_energy: Lowest energy anywhere in the trace. Biased low under shot noise, because
            it selects favourable fluctuations; kept for comparison with the 2022 challenge,
            which scored this number. Report ``final_energy`` instead.
        optimal_params: Final parameters.
        x0: Initial parameters.
        exact_energy: Reference ground-state energy.
        nfev: Objective evaluations.
        nit: Optimiser iterations.
        trace: Every evaluation.
        params_history: Parameters of every evaluation (if recorded).
        isa_stats: Transpiled-circuit statistics.
        estimator: Estimator description (backend, options).
        wall_time_s: Total optimisation time.
        optimizer_message: Termination reason.
        final_energy: The final parameters re-measured once with ``backend.final_shots``. This
            is the headline result.
        final_std: Standard error of ``final_energy`` as reported by the primitive (an upper
            bound for sampling estimators; 0 when exact).
        final_shots: Shots per measurement basis used for ``final_energy``; ``None`` if exact.
        total_shots: Circuit executions spent, including the re-measurement:
            ``sum(shots per evaluation) * measurement_groups``; 0 when exact.
        ideal_energy: Noise-free energy of the final parameters, if tracked. On noisy backends
            this measures how good the parameters are, separately from the noise floor.
        ideal_trace: Noise-free energy after every iteration, with shots spent, if tracked.
    """

    config: RunConfig
    energy: float
    best_energy: float
    optimal_params: FloatArray
    x0: FloatArray
    exact_energy: float
    nfev: int
    nit: int
    trace: list[TraceRow]
    params_history: list[FloatArray] = field(default_factory=list)
    isa_stats: dict[str, Any] = field(default_factory=dict)
    estimator: dict[str, Any] = field(default_factory=dict)
    wall_time_s: float = 0.0
    optimizer_message: str = ""
    final_energy: float = math.nan
    final_std: float = 0.0
    final_shots: int | None = None
    total_shots: int = 0
    ideal_energy: float | None = None
    ideal_trace: list[IdealPoint] = field(default_factory=list)

    @property
    def ideal_relative_error(self) -> float | None:
        """Relative error of the noise-free energy of the final parameters, if tracked."""
        return (
            None
            if self.ideal_energy is None
            else relative_error(self.exact_energy, self.ideal_energy)
        )

    @property
    def final_relative_error(self) -> float:
        """Relative error of the re-measured ``final_energy``: the number to report."""
        return relative_error(self.exact_energy, self.final_energy)

    @property
    def relative_error(self) -> float:
        """Relative error of ``energy`` against the exact value."""
        return relative_error(self.exact_energy, self.energy)

    @property
    def best_relative_error(self) -> float:
        """Relative error of ``best_energy`` against the exact value."""
        return relative_error(self.exact_energy, self.best_energy)

    def energies(self) -> FloatArray:
        """Energy per evaluation."""
        return np.asarray([r.energy for r in self.trace], dtype=float)

    def energies_per_iteration(self) -> FloatArray:
        """Lowest energy within each optimiser iteration (for iteration-axis plots)."""
        by_iter: dict[int, float] = {}
        for row in self.trace:
            by_iter[row.iteration] = min(by_iter.get(row.iteration, np.inf), row.energy)
        return np.asarray([by_iter[k] for k in sorted(by_iter)], dtype=float)


def prepare(cfg: RunConfig, *, settings: Settings | None = None) -> Prepared:
    """Build all objects and transpile, without running any circuit."""
    lattice = cfg.lattice.build()
    hamiltonian = heisenberg_hamiltonian(lattice)
    ansatz = cfg.ansatz.build(lattice)
    circuit = ansatz.build()
    if cfg.optimizer.uses_parameter_shift:
        circuit, parameter_map = unshare_parameters(circuit)
    else:
        parameter_map = ParameterMap.identity(ansatz.num_parameters)
    backend = resolve_backend(cfg.backend, settings)
    problem = compile_for_backend(
        circuit,
        hamiltonian,
        backend,
        optimization_level=cfg.backend.optimization_level,
        initial_layout=cfg.backend.initial_layout,
        seed_transpiler=cfg.seed,
    )
    spectrum = exact_spectrum(hamiltonian, k=3)
    optimizer = cfg.optimizer.build(np.random.default_rng(cfg.seed), parameter_map)
    per_iter = optimizer.evaluations_per_iteration(ansatz.num_parameters)
    projected = None if per_iter is None else per_iter * optimizer.maxiter + 1
    log.info(
        "prepared %s: %s on %s, %d parameters, exact E0=%.6f",
        cfg.name,
        ansatz.name,
        cfg.backend.label(),
        ansatz.num_parameters,
        spectrum[0],
    )
    return Prepared(
        config=cfg,
        lattice=lattice,
        hamiltonian=hamiltonian,
        ansatz=ansatz,
        problem=problem,
        backend=backend,
        exact_spectrum=spectrum,
        projected_evaluations=projected,
        parameter_map=parameter_map,
        twin=_twin(cfg, ansatz, hamiltonian, float(spectrum[0])),
    )


def _twin(
    cfg: RunConfig, ansatz: Ansatz, hamiltonian: SparsePauliOp, exact: float
) -> tuple[FloatArray, float] | None:
    """Twin parameters (ansatz space) and their noise-free energy, if rescaling is on."""
    rescale = cfg.mitigation.rescale
    if rescale is None:
        return None
    from kagome_vqe.mitigation import choose_twin

    circuit = ansatz.build()
    if rescale.twin == "auto":
        return choose_twin(circuit, hamiltonian, exact, rng=np.random.default_rng(cfg.seed))
    theta = np.asarray(rescale.twin, dtype=float)
    if theta.size != ansatz.num_parameters:
        msg = f"rescale.twin has {theta.size} values, the ansatz {ansatz.num_parameters}"
        raise ValueError(msg)
    with StatevectorEnergy(compile_for_backend(circuit, hamiltonian, None)) as est:
        return theta, float(est.evaluate(theta[None, :]).energies[0])


def _shot_cost(cfg: RunConfig, groups: int, measured_shots: int) -> int:
    """Circuit shots including mitigation overheads.

    ``measured_shots`` is the sum of per-basis shots over every energy requested. ZNE runs each
    energy once per scale factor; readout calibration costs two circuits per inner estimator
    (one per scale factor); the rescaling twin is measured once.
    """
    if cfg.backend.is_exact:
        return 0
    m = cfg.mitigation
    scales = len(m.zne.scales) if m.zne is not None else 1
    total = scales * groups * measured_shots
    if m.readout:
        total += scales * 2 * m.readout_shots
    if m.rescale is not None:
        twin_shots = m.rescale.shots or cfg.backend.final_shots or cfg.backend.shots or 0
        total += scales * groups * twin_shots
    return int(total)


def _ideal_energy_fn(prep: Prepared) -> Callable[[FloatArray], float] | None:
    """Noise-free energy of ansatz parameters, or ``None`` if tracking is off or too large."""
    if not prep.config.track_ideal_energy or prep.lattice.num_sites > IDEAL_TRACKING_MAX_QUBITS:
        return None
    logical = compile_for_backend(prep.ansatz.build(), prep.hamiltonian, None)
    estimator = StatevectorEnergy(logical)

    def ideal(theta: FloatArray) -> float:
        return float(estimator.evaluate(np.asarray(theta, dtype=float)[None, :]).energies[0])

    return ideal


def run_vqe(
    cfg: RunConfig,
    *,
    prepared: Prepared | None = None,
    settings: Settings | None = None,
    on_evaluation: Callable[[TraceRow], None] | None = None,
) -> VQERunResult:
    """Run the full VQE described by ``cfg``.

    Args:
        cfg: Run configuration.
        prepared: Output of :func:`prepare` if already available (avoids re-transpiling).
        settings: IBM credentials for ``backend.kind == "ibm"``.
        on_evaluation: Called after every energy evaluation (for progress display).

    Returns:
        The result, including the full evaluation trace.
    """
    prep = prepared or prepare(cfg, settings=settings)
    rng = np.random.default_rng(cfg.seed)
    x0 = cfg.x0_scale * rng.random(prep.ansatz.num_parameters)
    optimizer: Optimizer = cfg.optimizer.build(rng, prep.parameter_map)
    groups = 0 if cfg.backend.is_exact else prep.problem.measurement_groups
    ideal = _ideal_energy_fn(prep)
    ideal_trace: list[IdealPoint] = []

    start = time.perf_counter()
    with make_estimator(
        cfg.backend,
        prep.problem,
        prep.backend,
        seed=cfg.seed,
        mitigation=cfg.mitigation,
        twin=prep.circuit_twin,
    ) as estimator:
        objective = EnergyObjective(
            estimator, record_params=cfg.record_params, parameter_map=prep.parameter_map
        )

        def on_iteration(info: IterationInfo) -> None:
            objective.on_iteration(info)
            if ideal is not None:
                ideal_trace.append(
                    IdealPoint(
                        info.nit,
                        objective.nfev,
                        _shot_cost(cfg, groups, objective.shots_spent),
                        ideal(info.x),
                    )
                )

        if ideal is not None:
            ideal_trace.append(IdealPoint(0, 0, 0, ideal(x0)))
        if on_evaluation is not None:
            objective.add_listener(on_evaluation)
        log.info(
            "optimising with %s from |x0|=%.3f (%s evaluations projected)",
            optimizer.name,
            float(np.linalg.norm(x0)),
            "?" if prep.projected_evaluations is None else prep.projected_evaluations,
        )
        opt = optimizer.minimize(objective, x0, callback=on_iteration)
        final = estimator.evaluate(
            prep.parameter_map.expand(np.asarray(opt.x, dtype=float))[None, :],
            shots=cfg.backend.remeasure_shots,
        )
        description = estimator.description
    elapsed = time.perf_counter() - start

    total_shots = _shot_cost(cfg, groups, objective.shots_spent + (final.shots or 0))

    best = objective.best
    best_energy = opt.fun if best is None else min(opt.fun, best.energy)
    result = VQERunResult(
        config=cfg,
        energy=float(opt.fun),
        best_energy=float(best_energy),
        optimal_params=np.asarray(opt.x, dtype=float),
        x0=x0,
        exact_energy=prep.exact_energy,
        nfev=objective.nfev,
        nit=opt.nit,
        trace=objective.trace,
        params_history=objective.params,
        isa_stats=prep.problem.stats(),
        estimator=description,
        wall_time_s=elapsed,
        optimizer_message=opt.message,
        final_energy=float(final.energies[0]),
        final_std=float(final.stds[0]),
        final_shots=final.shots if not cfg.backend.is_exact else None,
        total_shots=int(total_shots),
        ideal_energy=None if ideal is None else ideal(opt.x),
        ideal_trace=ideal_trace,
    )
    log.info(
        "%s finished: E_final=%.6f (exact %.6f, rel. err %.2e), %d evaluations in %.1fs",
        cfg.name,
        result.final_energy,
        result.exact_energy,
        result.final_relative_error,
        result.nfev,
        elapsed,
    )
    return result


@dataclass(slots=True)
class FixedPointResult:
    """Repeated energy estimates at one fixed parameter point (no optimisation).

    Used to measure what a backend and mitigation stack reports for a state whose noise-free
    energy is known: the bias of the mean and the spread between repeats.

    Attributes:
        config: Run configuration (backend and mitigation).
        point: Ansatz parameters evaluated.
        ideal_energy: Noise-free energy at ``point``.
        exact_energy: Ground-state energy of the lattice.
        energies: One estimate per repeat, each with a fresh simulator seed.
        stds: Standard error each estimate reported.
        shots_per_evaluation: Circuit shots one estimate costs (all bases and scale factors).
        setup_shots: One-off shots per estimator (readout calibration, twin measurement); with a
            fresh setup per repeat they are spent once per repeat.
        description: Estimator description, including calibration and twin details.
        wall_time_s: Total time.
    """

    config: RunConfig
    point: FloatArray
    ideal_energy: float
    exact_energy: float
    energies: FloatArray
    stds: FloatArray
    shots_per_evaluation: int
    setup_shots: int
    description: dict[str, Any]
    wall_time_s: float

    @property
    def mean(self) -> float:
        """Mean estimate over repeats."""
        return float(self.energies.mean())

    @property
    def spread(self) -> float:
        """Sample standard deviation between repeats (the real statistical error)."""
        return float(self.energies.std(ddof=1)) if self.energies.size > 1 else 0.0

    @property
    def bias(self) -> float:
        """``mean - ideal_energy``: negative means the method overshoots below the truth."""
        return self.mean - self.ideal_energy

    @property
    def relative_bias(self) -> float:
        """``|bias| / |ideal_energy|``."""
        return abs(self.bias) / abs(self.ideal_energy)

    @property
    def below_ground(self) -> bool:
        """Whether the mean lies significantly below the exact ground energy.

        Significant means more than two standard errors of the mean. By the variational principle
        such a value is an artefact of the method, not a better answer.
        """
        sem = self.spread / np.sqrt(max(self.energies.size, 1))
        return bool(self.mean < self.exact_energy - 2 * sem)

    def summary(self) -> dict[str, Any]:
        """JSON-friendly summary."""
        return {
            "name": self.config.name,
            "ansatz": self.config.ansatz.type,
            "mitigation": self.config.mitigation.label(),
            "backend": self.config.backend.label(),
            "point": self.point.tolist(),
            "ideal_energy": self.ideal_energy,
            "exact_energy": self.exact_energy,
            "mean": self.mean,
            "spread": self.spread,
            "bias": self.bias,
            "relative_bias": self.relative_bias,
            "below_ground": self.below_ground,
            "kept_fraction": _find(self.description, "mean_kept_fraction"),
            "repeats": int(self.energies.size),
            "energies": self.energies.tolist(),
            "reported_stds": self.stds.tolist(),
            "shots_per_evaluation": self.shots_per_evaluation,
            "setup_shots": self.setup_shots,
            "wall_time_s": self.wall_time_s,
            "estimator": self.description,
        }


def _find(tree: Any, key: str) -> Any:
    """First value stored under ``key`` anywhere in nested estimator descriptions, else None."""
    if isinstance(tree, dict):
        if tree.get(key) is not None:
            return tree[key]
        for value in tree.values():
            found = _find(value, key)
            if found is not None:
                return found
    return None


def evaluate_fixed(
    cfg: RunConfig,
    point: float | FloatArray,
    *,
    repeats: int = 10,
    fresh_setup: bool = True,
    prepared: Prepared | None = None,
    settings: Settings | None = None,
) -> FixedPointResult:
    """Estimate the energy at a fixed point ``repeats`` times through the configured stack.

    Args:
        cfg: Run configuration; the optimiser settings are ignored.
        point: Ansatz parameters, or one value used for every parameter.
        repeats: Independent estimates.
        fresh_setup: Build a new estimator, with its own seed, for every repeat, so that one-off
            measurements (readout calibration, twin) are redone each time. Then the spread
            between repeats includes their noise, which a shared setup would hide: a twin
            measured once shifts every repeat by the same amount. ``False`` shares one setup
            and only resamples the energy itself.
        prepared: Output of :func:`prepare`, if available.
        settings: IBM credentials for ``backend.kind == "ibm"``.
    """
    prep = prepared or prepare(cfg, settings=settings)
    n = prep.ansatz.num_parameters
    theta = (
        np.full(n, float(point))
        if np.ndim(point) == 0
        else np.asarray(point, dtype=float).reshape(n)
    )
    logical = compile_for_backend(prep.ansatz.build(), prep.hamiltonian, None)
    with StatevectorEnergy(logical) as sv:
        ideal = float(sv.evaluate(theta[None, :]).energies[0])
    groups = 0 if cfg.backend.is_exact else prep.problem.measurement_groups
    phi = prep.parameter_map.expand(theta)[None, :]
    seeds = np.random.default_rng(cfg.seed).integers(0, 2**31 - 1, size=repeats)

    def open_estimator(seed: int) -> EnergyEstimator:
        return make_estimator(
            cfg.backend,
            prep.problem,
            prep.backend,
            seed=seed,
            mitigation=cfg.mitigation,
            twin=prep.circuit_twin,
        )

    start = time.perf_counter()
    energies: list[float] = []
    stds: list[float] = []
    setups: list[dict[str, Any]] = []
    if fresh_setup:
        for seed in seeds:
            with open_estimator(int(seed)) as estimator:
                batch = estimator.evaluate(phi)
                setups.append(estimator.description)
            energies.append(float(batch.energies[0]))
            stds.append(float(batch.stds[0]))
    else:
        with open_estimator(int(seeds[0])) as estimator:
            for _ in range(repeats):
                batch = estimator.evaluate(phi)
                energies.append(float(batch.energies[0]))
                stds.append(float(batch.stds[0]))
            setups.append(estimator.description)
    shots = cfg.backend.shots or 0
    per_eval = _shot_cost(cfg, groups, shots) - _shot_cost(cfg, groups, 0)
    description = dict(setups[-1])
    description["fresh_setup_per_repeat"] = fresh_setup
    twin_energies = [_find(d, "noisy_twin_energy") for d in setups]
    if any(e is not None for e in twin_energies):
        description["noisy_twin_energy_per_setup"] = twin_energies
    return FixedPointResult(
        config=cfg,
        point=theta,
        ideal_energy=ideal,
        exact_energy=prep.exact_energy,
        energies=np.asarray(energies),
        stds=np.asarray(stds),
        shots_per_evaluation=per_eval,
        setup_shots=_shot_cost(cfg, groups, 0),
        description=description,
        wall_time_s=time.perf_counter() - start,
    )
