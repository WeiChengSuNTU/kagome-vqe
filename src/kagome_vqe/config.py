"""Run configuration: typed models, TOML loading, command-line overrides, and secrets.

A :class:`RunConfig` fully determines one VQE run. It is written to every results directory so
a run can be replayed with ``kagome-vqe run results/<dir>/config.toml``. Each sub-model has a
``build()`` method that turns the declarative description into the corresponding object
(lattice, ansatz, optimiser), so the models are the single place where names such as
``"hva"`` are mapped to classes.

Credentials never live in configs. :class:`Settings` reads them from the environment or a
git-ignored ``.env`` file.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from kagome_vqe.ansatz import Ansatz, GraphHVAnsatz, HVAnsatz, LibraryAnsatz, RVBAnsatz
from kagome_vqe.ansatz.base import ring_bonds
from kagome_vqe.ansatz.unshare import ParameterMap
from kagome_vqe.kagome_lattices import (
    custom_lattice,
    kagome_bm_rectangle,
    kagome_kvqe20,
    kagome_patch,
    kagome_periodic,
)
from kagome_vqe.lattice import (
    KAGOME_HAMILTONIAN_CYCLE,
    Lattice,
    find_dimer_covering,
    kagome_unit_cell,
)
from kagome_vqe.optimizers import (
    Adam,
    GradientDescent,
    Optimizer,
    ScipyOptimizer,
    Spsa,
    make_gradient,
)
from kagome_vqe.schedule import make_schedule

__all__ = [
    "AnsatzConfig",
    "BackendConfig",
    "BackendKind",
    "BenchmarkConfig",
    "BudgetConfig",
    "ComboConfig",
    "LatticeConfig",
    "LatticeKind",
    "MappingConfig",
    "MitigationConfig",
    "OptimizerConfig",
    "RescaleConfig",
    "RunConfig",
    "Settings",
    "ZneConfig",
    "apply_overrides",
    "load_benchmark_config",
    "load_run_config",
    "load_toml",
    "parse_override",
]

BackendKind = Literal["statevector", "aer_ideal", "aer_noisy", "fake", "ibm"]
"""Where energies are evaluated.

``statevector``
    Exact expectation values of the logical circuit (no transpilation, no shots). Fast; the
    reference for everything else.
``aer_ideal``
    Aer shot-based simulation without noise.
``aer_noisy``
    Aer with the noise model and coupling map of the fake backend ``name``. Fast enough for the
    benchmark; this is what "FakeXXX simulator" means in practice.
``fake``
    The fake backend ``name`` driven through ``qiskit_ibm_runtime``'s local testing mode, i.e.
    exactly the code path used for ``ibm`` (options, resilience levels, sessions) minus the
    network. Slower than ``aer_noisy``; use it to rehearse a hardware run.
``ibm``
    A real IBM Quantum device ``name`` via the cloud (credentials from :class:`Settings`).
"""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


LatticeKind = Literal[
    "kagome_unit_cell",
    "kagome_patch",
    "kagome_kvqe20",
    "kagome_bm_rectangle",
    "kagome_periodic",
    "custom",
]
"""Lattice generators, see :mod:`kagome_vqe.kagome_lattices`.

``kagome_unit_cell``
    The 12-site star (default; the original problem).
``kagome_patch``
    Open patch on a ``rows x cols`` window of the triangular lattice (minus ``drop``).
``kagome_kvqe20``
    The 20-site open patch of PRB 106, Fig. 2.
``kagome_bm_rectangle``
    The kagome graph on a ``rows x cols`` qubit rectangle of the PRB 105 embedding.
``kagome_periodic``
    ``rows x cols`` torus of unit cells (``3 rows cols`` sites).
``custom``
    Explicit ``num_sites`` and ``bonds`` (optionally ``coords``, ``boundary``, ``dimers``).
"""

_SIZED_KINDS = ("kagome_patch", "kagome_bm_rectangle", "kagome_periodic")
_CUSTOM_ONLY = ("num_sites", "bonds", "coords", "wrapped_bonds")


class LatticeConfig(_Frozen):
    """Which spin graph to solve.

    Attributes:
        kind: Generator, see :data:`LatticeKind`.
        t: Default exchange coupling (every bond of a generated lattice; two-element custom
            bonds).
        site_order: How ansatz qubits map to lattice sites. ``None`` (default) means
            ``"hamiltonian_cycle"`` for the unit cell (which makes the ring ansatzes exact) and
            ``"identity"`` otherwise; or an explicit permutation.
        rows: Size for ``kagome_patch`` / ``kagome_bm_rectangle`` / ``kagome_periodic``.
        cols: Size for the same kinds.
        drop: ``kagome_patch`` only: coordinates ``[r, c]`` to leave out.
        num_sites: ``custom`` only.
        bonds: ``custom`` only: ``[a, b]`` or ``[a, b, J_ab]`` per bond.
        coords: ``custom`` only: triangular-lattice coordinates ``[r, c]`` per site (needed by
            the geometric mappings).
        boundary: ``custom`` only (generated lattices know their boundary): ``"open"``,
            ``"periodic"`` or ``"unspecified"``.
        wrapped_bonds: ``custom`` only: bonds crossing a periodic boundary.
        dimers: Reference dimer covering for ``graph_hva``: ``"default"`` (the generator's
            covering if it has one, else the deterministic first perfect matching), ``"auto"``
            (always the deterministic first perfect matching) or explicit ``[a, b]`` pairs.
    """

    kind: LatticeKind = "kagome_unit_cell"
    t: float = 1.0
    site_order: Literal["hamiltonian_cycle", "identity"] | list[int] | None = None
    rows: Annotated[int, Field(ge=1)] | None = None
    cols: Annotated[int, Field(ge=1)] | None = None
    drop: list[tuple[int, int]] = Field(default_factory=list)
    num_sites: Annotated[int, Field(ge=1)] | None = None
    bonds: list[list[float]] | None = None
    coords: list[tuple[int, int]] | None = None
    boundary: Literal["open", "periodic", "unspecified"] | None = None
    wrapped_bonds: list[tuple[int, int]] = Field(default_factory=list)
    dimers: Literal["default", "auto"] | list[tuple[int, int]] = "default"

    @model_validator(mode="after")
    def _fields_fit_kind(self) -> Self:
        if self.kind in _SIZED_KINDS and (self.rows is None or self.cols is None):
            msg = f"lattice.kind={self.kind!r} needs lattice.rows and lattice.cols"
            raise ValueError(msg)
        if self.kind not in _SIZED_KINDS and (self.rows is not None or self.cols is not None):
            msg = f"lattice.rows/cols do not apply to kind={self.kind!r}"
            raise ValueError(msg)
        if self.drop and self.kind != "kagome_patch":
            msg = "lattice.drop only applies to kind='kagome_patch'"
            raise ValueError(msg)
        if self.kind == "custom":
            if self.num_sites is None or self.bonds is None:
                msg = "lattice.kind='custom' needs lattice.num_sites and lattice.bonds"
                raise ValueError(msg)
        else:
            given = [f for f in _CUSTOM_ONLY if getattr(self, f) not in (None, [])]
            if given or self.boundary is not None:
                extra = given + (["boundary"] if self.boundary is not None else [])
                msg = f"lattice fields {extra} only apply to kind='custom'"
                raise ValueError(msg)
        if self.site_order == "hamiltonian_cycle" and self.kind != "kagome_unit_cell":
            msg = "site_order='hamiltonian_cycle' is only defined for the kagome unit cell"
            raise ValueError(msg)
        return self

    def _generate(self) -> Lattice:
        if self.kind == "kagome_unit_cell":
            return kagome_unit_cell(self.t, site_order=None)
        if self.kind == "kagome_kvqe20":
            return kagome_kvqe20(self.t)
        if self.kind == "custom":
            assert self.num_sites is not None
            assert self.bonds is not None
            return custom_lattice(
                self.num_sites,
                self.bonds,
                coupling=self.t,
                coords=self.coords,
                boundary=self.boundary or "unspecified",
                wrapped_bonds=self.wrapped_bonds,
            )
        assert self.rows is not None
        assert self.cols is not None
        if self.kind == "kagome_patch":
            return kagome_patch(self.rows, self.cols, self.t, drop=self.drop)
        if self.kind == "kagome_bm_rectangle":
            return kagome_bm_rectangle(self.rows, self.cols, self.t)
        return kagome_periodic(self.rows, self.cols, self.t)

    def build(self) -> Lattice:
        """Construct and validate the lattice (site order and dimers applied)."""
        lattice = self._generate()
        if self.site_order is None:
            order = KAGOME_HAMILTONIAN_CYCLE if self.kind == "kagome_unit_cell" else None
        elif self.site_order == "hamiltonian_cycle":
            order = KAGOME_HAMILTONIAN_CYCLE
        elif self.site_order == "identity":
            order = None
        else:
            order = tuple(self.site_order)
        lattice = lattice.with_site_order(order)
        if self.dimers == "auto":
            lattice = lattice.with_dimers(find_dimer_covering(lattice))
        elif self.dimers != "default":
            lattice = lattice.with_dimers([(int(a), int(b)) for a, b in self.dimers])
        return lattice


_GRAPH_HVA_ONLY = (
    "parameterization",
    "include_h0",
    "h0_position",
    "prune_phase_gates",
    "schedule",
    "layers",
    "layer_order",
)
_GRAPH_HVA_DEFAULTS: dict[str, Any] = {
    "parameterization": "opg",
    "include_h0": False,
    "h0_position": "end",
    "prune_phase_gates": False,
    "schedule": "greedy",
    "layers": None,
    "layer_order": None,
}


class AnsatzConfig(_Frozen):
    """Which trial circuit to use.

    Attributes:
        type: ``"rvb"``, ``"hva"`` (ring ansatzes), ``"graph_hva"`` (graph-based HVA),
            ``"real_amplitudes"`` or ``"efficient_su2"``.
        reps: Number of layers / HVA cycles ``p``.
        gate: RVB only: ``"pauli_evolution"`` (true eSWAP) or ``"legacy"`` (original gate).
        lattice_aware: HVA only: add a third sub-layer on the lattice bonds that are not on
            the ansatz ring.
        entanglement: Library ansatzes only: Qiskit entanglement pattern.
        parameterization: graph_hva: ``"opg"`` (one parameter per gate, default) or
            ``"per_layer"`` (one per schedule layer and cycle, explicit sharing).
        include_h0: graph_hva: add an independently parameterised H0 evolution per cycle.
        h0_position: graph_hva: ``"end"`` (default; PRB 105 Eq. (4) read as an operator
            product) or ``"start"`` of each cycle. At the start of cycle 1 it is a pure phase.
        prune_phase_gates: graph_hva: drop gates that only add a global phase.
        schedule: graph_hva gate order: ``"greedy"``, ``"prb105"``, ``"prb105_fig3"``,
            ``"prb106"`` or ``"explicit"`` (see :mod:`kagome_vqe.schedule`).
        layers: graph_hva with ``schedule = "explicit"``: bond layers in site labels.
        layer_order: graph_hva: optional permutation applied to the schedule's layers.
    """

    type: Literal["rvb", "hva", "graph_hva", "real_amplitudes", "efficient_su2"] = "hva"
    reps: Annotated[int, Field(ge=1)] = 1
    gate: Literal["pauli_evolution", "legacy"] = "pauli_evolution"
    lattice_aware: bool = False
    entanglement: str = "reverse_linear"
    parameterization: Literal["opg", "per_layer"] = "opg"
    include_h0: bool = False
    h0_position: Literal["start", "end"] = "end"
    prune_phase_gates: bool = False
    schedule: Literal["greedy", "prb105", "prb105_fig3", "prb106", "explicit"] = "greedy"
    layers: list[list[tuple[int, int]]] | None = None
    layer_order: list[int] | None = None

    @model_validator(mode="after")
    def _graph_options(self) -> Self:
        if self.type != "graph_hva":
            changed = [f for f in _GRAPH_HVA_ONLY if getattr(self, f) != _GRAPH_HVA_DEFAULTS[f]]
            if changed:
                msg = f"ansatz fields {changed} only apply to type='graph_hva'"
                raise ValueError(msg)
        if (self.schedule == "explicit") != (self.layers is not None):
            msg = "ansatz.layers is required for, and only allowed with, schedule='explicit'"
            raise ValueError(msg)
        return self

    def build(self, lattice: Lattice) -> Ansatz:
        """Construct the ansatz for ``lattice.num_sites`` qubits."""
        n = lattice.num_sites
        if self.type == "graph_hva":
            dimers = lattice.dimers if lattice.dimers is not None else find_dimer_covering(lattice)
            schedule = make_schedule(
                lattice,
                self.schedule,
                dimers=dimers,
                layers=self.layers,
                include_h0=self.include_h0,
            )
            if self.layer_order is not None:
                schedule = schedule.reordered(self.layer_order)
            return GraphHVAnsatz(
                lattice,
                self.reps,
                schedule=schedule,
                dimers=dimers,
                include_h0=self.include_h0,
                h0_position=self.h0_position,
                parameterization=self.parameterization,
                prune_phase_gates=self.prune_phase_gates,
            )
        if self.type == "rvb":
            return RVBAnsatz(n, self.reps, gate=self.gate)
        if self.type == "hva":
            extra: list[tuple[int, int]] = []
            if self.lattice_aware:
                odd, even = ring_bonds(n)
                ring = {frozenset(b) for b in odd + even}
                extra = [
                    (a, b) for a, b, _ in lattice.qubit_edges() if frozenset((a, b)) not in ring
                ]
            return HVAnsatz(n, self.reps, extra_bonds=extra)
        return LibraryAnsatz(self.type, n, self.reps, self.entanglement)


class MappingConfig(_Frozen):
    """How the logical circuit is embedded into hardware connectivity.

    Attributes:
        strategy: ``"generic"`` (default: Qiskit layout + SABRE routing, exactly the previous
            behaviour for the ring ansatzes), ``"prb105"`` (Bosse-Montanaro square-grid
            embedding) or ``"prb106_grid"`` (Kattemolle-van Wezel grid with swapping stations).
            See :mod:`kagome_vqe.mapping.strategies`.
        fallback: ``"error"`` (default) fails when a geometric strategy does not apply (no
            coordinates, periodic boundary, no fit on the device); ``"generic"`` falls back to
            Qiskit routing and records ``status = "generic_fallback"``.
        physical_layout: Geometric strategies: hardware qubit per routed-circuit qubit
            (order of ``mapping.grid_positions`` in a dry run); ``None`` searches the coupling
            map and picks the embedding with the lowest calibrated error.
        restore_at_end: Undo outstanding SWAPs at the end (identity final permutation).
        merge_swaps: Fold a SWAP into an adjacent exchange gate on the same pair.
        exchange_synthesis: ``"cx3"`` (3-CNOT Heisenberg gate, default) or ``"rotations"``
            (``rzz ryy rxx``, two CNOTs each).
    """

    strategy: Literal["generic", "prb105", "prb106_grid"] = "generic"
    fallback: Literal["error", "generic"] = "error"
    physical_layout: list[int] | None = None
    restore_at_end: bool = False
    merge_swaps: bool = True
    exchange_synthesis: Literal["cx3", "rotations"] = "cx3"

    @model_validator(mode="after")
    def _layout_needs_geometry(self) -> Self:
        if self.physical_layout is not None and self.strategy == "generic":
            msg = "mapping.physical_layout is for geometric strategies; use backend.initial_layout"
            raise ValueError(msg)
        return self

    def label(self) -> str:
        """Short name for file names."""
        return self.strategy


_SCIPY_GRADIENT_METHODS = frozenset(
    {"L-BFGS-B", "BFGS", "CG", "SLSQP", "TNC", "TRUST-CONSTR", "NEWTON-CG"}
)


class OptimizerConfig(_Frozen):
    """Classical optimiser settings.

    Attributes:
        type: ``"adam"``, ``"gd"`` (plain gradient descent), ``"spsa"`` or ``"scipy"``.
        maxiter: Iteration budget.
        lr: ADAM / gradient-descent learning rate, SPSA step size.
        perturbation: SPSA perturbation magnitude, also used by ``gradient = "spsa"``.
        fd_eps: Finite-difference step (use ~0.1 under shot noise).
        gradient: Gradient estimator for ``adam`` and ``gd``: ``"forward"`` or ``"central"``
            differences, ``"spsa"`` (two-point simultaneous perturbation; with ``adam`` this is
            the SPSA-ADAM hybrid) or ``"parameter_shift"`` (exact up to shot noise; the circuit
            is compiled with one parameter per gate). Ignored by ``spsa`` and ``scipy``.
        calibrate: SPSA: derive decaying gains automatically instead of fixed ``lr``.
        method: scipy method name.
        tol: Convergence tolerance (ADAM update norm / scipy ``tol``).
        beta1: ADAM first-moment decay.
        beta2: ADAM second-moment decay.
        amsgrad: ADAM: use the AMSGrad variant.
        alpha: Calibrated SPSA learning-rate decay exponent (Spall's 0.602).
        gamma: Calibrated SPSA perturbation decay exponent (Spall's 0.101).
        stability_constant: Calibrated SPSA ``A``; ``None`` means ``0.1 * maxiter``.
        calibration_samples: Calibrated SPSA gradient probes (each costs 2 evaluations, which
            are counted in the run's evaluations and shots).
        target_magnitude: Calibrated SPSA first-step size; ``None`` means ``2 pi / 10``.
        options: Extra scipy ``options`` (for example ``eps`` for L-BFGS-B's finite-difference
            step, which must be far above ``1e-8`` with shot-noisy energies, or ``maxfun``).
    """

    type: Literal["adam", "gd", "spsa", "scipy"] = "adam"
    maxiter: Annotated[int, Field(ge=1)] = 100
    lr: Annotated[float, Field(gt=0)] = 0.01
    perturbation: Annotated[float, Field(gt=0)] = 0.01
    fd_eps: Annotated[float, Field(gt=0)] = 1e-6
    gradient: Literal["forward", "central", "spsa", "parameter_shift"] = "forward"
    calibrate: bool = False
    method: str = "COBYLA"
    tol: float | None = None
    beta1: Annotated[float, Field(ge=0, lt=1)] = 0.9
    beta2: Annotated[float, Field(ge=0, lt=1)] = 0.99
    amsgrad: bool = False
    alpha: Annotated[float, Field(gt=0)] = 0.602
    gamma: Annotated[float, Field(gt=0)] = 0.101
    stability_constant: Annotated[float, Field(ge=0)] | None = None
    calibration_samples: Annotated[int, Field(ge=1)] = 25
    target_magnitude: Annotated[float, Field(gt=0)] | None = None
    options: dict[str, float | int | bool | str] = Field(default_factory=dict)

    @property
    def uses_parameter_shift(self) -> bool:
        """Whether the circuit must be compiled with one parameter per gate occurrence."""
        return self.type in ("adam", "gd") and self.gradient == "parameter_shift"

    def label(self) -> str:
        """Short name for file names: ``adam``, ``adam_parameter_shift``, ``spsa``, ..."""
        if self.type in ("adam", "gd") and (self.type == "gd" or self.gradient != "forward"):
            return f"{self.type}_{self.gradient}"
        if self.type == "scipy":
            return f"scipy_{self.method.lower()}"
        return self.type

    def build(
        self, rng: np.random.Generator, parameter_map: ParameterMap | None = None
    ) -> Optimizer:
        """Construct the optimiser.

        Args:
            rng: Seeds SPSA's random directions (also for ``gradient = "spsa"``).
            parameter_map: Occurrence map of the compiled circuit, required for
                ``gradient = "parameter_shift"`` when parameters are shared.
        """
        if self.type in ("adam", "gd"):
            gradient = make_gradient(
                self.gradient,
                fd_eps=self.fd_eps,
                perturbation=self.perturbation,
                rng=rng,
                parameter_map=parameter_map,
            )
            if self.type == "gd":
                return GradientDescent(maxiter=self.maxiter, lr=self.lr, gradient=gradient)
            return Adam(
                maxiter=self.maxiter,
                lr=self.lr,
                beta1=self.beta1,
                beta2=self.beta2,
                gradient=gradient,
                tol=1e-6 if self.tol is None else self.tol,
                amsgrad=self.amsgrad,
            )
        if self.type == "spsa":
            extra = (
                {} if self.target_magnitude is None else {"target_magnitude": self.target_magnitude}
            )
            return Spsa(
                maxiter=self.maxiter,
                learning_rate=self.lr,
                perturbation=self.perturbation,
                calibrate=self.calibrate,
                alpha=self.alpha,
                gamma=self.gamma,
                stability_constant=self.stability_constant,
                calibration_samples=self.calibration_samples,
                seed=rng,
                **extra,
            )
        return ScipyOptimizer(
            method=self.method, maxiter=self.maxiter, tol=self.tol, options=self.options
        )

    def gradient_description(self) -> str:
        """The gradient information the optimiser actually uses, for result files.

        Recorded with every run so that, for example, ``adam`` with the default forward
        difference can never be mistaken for the ADAM + SPSA-gradient hybrid.
        """
        if self.type in ("adam", "gd"):
            detail = {
                "forward": f"forward difference (step {self.fd_eps:g}), n + 1 energies",
                "central": f"central difference (step {self.fd_eps:g}), 2n + 1 energies",
                "spsa": f"simultaneous perturbation (c = {self.perturbation:g}), 2 energies",
                "parameter_shift": "parameter shift, 2 energies per gate occurrence",
            }[self.gradient]
            return f"{self.gradient}: {detail}"
        if self.type == "spsa":
            schedule = (
                f"calibrated gains (alpha {self.alpha:g}, gamma {self.gamma:g}, "
                f"{self.calibration_samples} calibration probes)"
                if self.calibrate
                else "constant gains"
            )
            return f"spsa: built-in two-point estimate (c = {self.perturbation:g}), {schedule}"
        if self.method.upper() in _SCIPY_GRADIENT_METHODS:
            eps = self.options.get("eps", "scipy default 1e-8")
            return f"scipy 2-point finite difference (eps = {eps}), n + 1 energies per gradient"
        return "none (derivative-free)"


class BackendConfig(_Frozen):
    """Where and how energies are evaluated. See :data:`BackendKind`.

    Attributes:
        kind: Evaluation backend family.
        name: Fake backend class (``"FakeGuadalupeV2"``) or IBM device (``"ibm_torino"``).
            Required for ``aer_noisy``, ``fake`` and ``ibm``.
        shots: Shots per measurement basis for every energy the optimiser requests. Required
            for every kind except ``statevector``, where it is ignored.
        final_shots: Shots for the single re-measurement of the optimiser's final parameters.
            This, not the minimum over the trace, is the reported energy: the minimum of noisy
            samples is biased low. ``None`` reuses ``shots``.
        noise: ``aer_noisy`` only: which parts of the fake backend's noise model to apply,
            ``"full"``, ``"readout"`` (measurement errors only) or ``"gates"`` (gate errors and
            T1/T2, no readout errors). Used to see how mitigation acts on each error source.
        aer_method: Aer simulation method for ``aer_ideal``/``aer_noisy``. ``"auto"`` uses
            density matrices up to 12 active qubits (cost independent of shots) and statevector
            trajectories beyond.
        resilience_level: Runtime error-mitigation level (0-2); ``fake``/``ibm`` only.
        optimization_level: Transpiler preset level.
        initial_layout: Physical qubits for the ansatz qubits; ``None`` lets the transpiler
            choose. For Guadalupe use :data:`~kagome_vqe.lattice.GUADALUPE_HEAVY_HEX_LOOP`.
        mode: Runtime execution mode for ``ibm``/``fake``: ``"session"`` keeps the device
            reserved across iterations (recommended for VQE), ``"batch"`` or single ``"job"``.
        max_time: Session/batch time limit (``"2h"``, ``"90m"`` or seconds).
        estimator_impl: ``"legacy_v2"`` (``qiskit_ibm_runtime.EstimatorV2``, deprecated in
            0.50) or ``"executor"`` (its client-side successor).
        retries: How many times a failed/timed-out runtime job is resubmitted.
        job_timeout_s: Seconds to wait for a single runtime job's result.
    """

    kind: BackendKind = "statevector"
    name: str | None = None
    shots: Annotated[int, Field(ge=1)] | None = 4096
    final_shots: Annotated[int, Field(ge=1)] | None = 16384
    resilience_level: Annotated[int, Field(ge=0, le=2)] = 0
    optimization_level: Annotated[int, Field(ge=0, le=3)] = 3
    initial_layout: list[int] | None = None
    mode: Literal["job", "session", "batch"] = "session"
    max_time: str | None = "2h"
    estimator_impl: Literal["legacy_v2", "executor"] = "legacy_v2"
    retries: Annotated[int, Field(ge=0)] = 5
    job_timeout_s: Annotated[int, Field(ge=1)] = 3600
    aer_method: Literal["auto", "density_matrix", "statevector", "matrix_product_state"] = "auto"
    noise: Literal["full", "readout", "gates"] = "full"

    @model_validator(mode="after")
    def _needs_name_and_shots(self) -> Self:
        if self.kind in ("aer_noisy", "fake", "ibm") and not self.name:
            msg = f"backend.kind={self.kind!r} requires backend.name"
            raise ValueError(msg)
        if self.noise != "full" and self.kind != "aer_noisy":
            msg = "backend.noise only applies to kind = 'aer_noisy'"
            raise ValueError(msg)
        if self.kind != "statevector" and self.shots is None:
            msg = f"backend.kind={self.kind!r} samples measurements and requires backend.shots"
            raise ValueError(msg)
        return self

    @property
    def remeasure_shots(self) -> int | None:
        """Shots for the final re-measurement (``None`` for exact evaluation)."""
        if self.is_exact:
            return None
        return self.final_shots or self.shots

    @property
    def is_exact(self) -> bool:
        """Whether energies are noise- and shot-free."""
        return self.kind == "statevector"

    @property
    def uses_runtime(self) -> bool:
        """Whether the qiskit-ibm-runtime primitives are used."""
        return self.kind in ("fake", "ibm")

    def label(self) -> str:
        """Short label for file names, e.g. ``"aer_noisy:FakeGuadalupeV2"``."""
        base = self.kind if self.name is None else f"{self.kind}:{self.name}"
        return base if self.noise == "full" else f"{base}[{self.noise} noise]"


class ZneConfig(_Frozen):
    """Zero-noise extrapolation settings.

    Attributes:
        scales: Odd noise-scale factors; each two-qubit gate is applied that many times.
        extrapolation: ``"exponential"`` (matches multiplicative damping), ``"linear"``,
            ``"quadratic"`` or ``"richardson"``.
    """

    scales: list[Annotated[int, Field(ge=1)]] = Field(default_factory=lambda: [1, 3, 5])
    extrapolation: Literal["linear", "quadratic", "richardson", "exponential"] = "exponential"

    @field_validator("scales")
    @classmethod
    def _odd_scales(cls, value: list[int]) -> list[int]:
        if len(value) < 2 or any(s % 2 == 0 for s in value):
            msg = "zne.scales needs at least two odd integers"
            raise ValueError(msg)
        return value


class RescaleConfig(_Frozen):
    """Twin-circuit rescaling settings.

    Attributes:
        twin: ``"auto"`` searches multiples of ``pi/4`` for the non-ground twin with the largest
            noise-free energy; or explicit ansatz parameters.
        shots: Shots for the one-off twin measurement; ``None`` uses ``backend.final_shots``.
    """

    twin: Literal["auto"] | list[float] = "auto"
    shots: Annotated[int, Field(ge=1)] | None = None


class MitigationConfig(_Frozen):
    """Error mitigation, applied in this order: counts-level, then ZNE, then rescaling.

    Attributes:
        readout: Calibrate per-qubit readout errors and correct every shot.
        readout_shots: Shots per readout-calibration circuit (two circuits, once per run).
        symmetry: Keep only shots in the total-spin-singlet sector (``n/2`` ones in every
            basis). Valid for the RVB and HVA ansatzes only.
        zne: Zero-noise extrapolation, or ``None``.
        rescale: Twin-circuit rescaling, or ``None``.
    """

    readout: bool = False
    readout_shots: Annotated[int, Field(ge=1)] = 16384
    symmetry: bool = False
    zne: ZneConfig | None = None
    rescale: RescaleConfig | None = None

    @property
    def enabled(self) -> bool:
        """Whether any technique is switched on."""
        return self.readout or self.symmetry or self.zne is not None or self.rescale is not None

    @property
    def needs_counts(self) -> bool:
        """Whether shots must be measured directly (readout or symmetry)."""
        return self.readout or self.symmetry

    def label(self) -> str:
        """Short name, e.g. ``"readout+symmetry+zne"``; ``"none"`` when disabled."""
        parts = [
            name
            for name, on in (
                ("readout", self.readout),
                ("symmetry", self.symmetry),
                ("zne", self.zne is not None),
                ("twin", self.rescale is not None),
            )
            if on
        ]
        return "+".join(parts) or "none"


class BudgetConfig(_Frozen):
    """Hard resource limits for the optimisation, enforced on every energy request.

    A batch that would take the run past a limit is refused before it is evaluated; the run
    then ends with the parameters of the last completed optimiser iteration and the
    termination reason ``"budget exhausted"``. Everything the optimiser requests counts,
    including SPSA calibration probes and the optimisers' own final evaluation. The single
    re-measurement of the final parameters (``backend.final_shots``) is outside the budget but
    is included in the run's reported ``total_shots``, the same for every optimiser.

    Attributes:
        max_evaluations: Energy evaluations (one parameter vector each).
        max_shots: Circuit shots over all measurement bases (shot-based backends only).
    """

    max_evaluations: Annotated[int, Field(ge=1)] | None = None
    max_shots: Annotated[int, Field(ge=1)] | None = None

    @property
    def enabled(self) -> bool:
        """Whether any limit is set."""
        return self.max_evaluations is not None or self.max_shots is not None


class RunConfig(_Frozen):
    """Everything needed for one reproducible VQE run.

    Attributes:
        name: Run label (used in the results directory name).
        seed: Master seed; fans out to the initial point, SPSA, transpiler and simulators.
        x0_scale: Initial parameters are drawn uniformly from ``[0, x0_scale)`` (the original
            used ``pi / 4``). ``0`` starts at the dimer state, which is already optimal with
            the default site order.
        lattice: See :class:`LatticeConfig`.
        ansatz: See :class:`AnsatzConfig`.
        optimizer: See :class:`OptimizerConfig`.
        backend: See :class:`BackendConfig`.
        output_dir: Root under which the results directory is created.
        record_params: Store the parameter vector of every evaluation (``params.npy``).
        track_ideal_energy: After every iteration, compute the noise-free (statevector) energy
            of the current parameters and store it with the cumulative shot count. This
            measures parameter quality independently of noise, which is the fair way to
            compare optimisers on noisy backends. Classical cost; skipped above 20 qubits.
        mitigation: Error mitigation, see :class:`MitigationConfig`.
        mapping: Hardware mapping, see :class:`MappingConfig`.
        budget: Resource limits, see :class:`BudgetConfig`.
    """

    name: str = "run"
    seed: int = 42
    x0_scale: Annotated[float, Field(ge=0)] = math.pi / 4
    lattice: LatticeConfig = LatticeConfig()
    ansatz: AnsatzConfig = AnsatzConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    backend: BackendConfig = BackendConfig()
    output_dir: Path = Path("results")
    record_params: bool = True
    track_ideal_energy: bool = True
    mitigation: MitigationConfig = MitigationConfig()
    mapping: MappingConfig = MappingConfig()
    budget: BudgetConfig = BudgetConfig()

    @model_validator(mode="after")
    def _mitigation_fits(self) -> Self:
        if self.mitigation.enabled and self.backend.kind == "statevector":
            msg = "mitigation needs a shot-based backend; statevector energies are exact"
            raise ValueError(msg)
        if self.mitigation.symmetry and self.ansatz.type not in ("rvb", "hva", "graph_hva"):
            msg = "symmetry filtering assumes a total-spin singlet ansatz (rvb, hva or graph_hva)"
            raise ValueError(msg)
        if self.mapping.strategy != "generic" and self.backend.initial_layout is not None:
            msg = "backend.initial_layout is for generic routing; use mapping.physical_layout"
            raise ValueError(msg)
        if self.budget.max_shots is not None and self.backend.is_exact:
            msg = "budget.max_shots needs a shot-based backend; use budget.max_evaluations"
            raise ValueError(msg)
        return self

    @field_validator("name")
    @classmethod
    def _safe_name(cls, value: str) -> str:
        if not value or any(c in value for c in "/\\ \t\n"):
            msg = "name must be non-empty and contain no whitespace or path separators"
            raise ValueError(msg)
        return value

    def to_toml_dict(self) -> dict[str, Any]:
        """JSON/TOML-safe dictionary (paths and secrets stringified)."""
        data = self.model_dump(mode="json")
        return _drop_none(data)


class ComboConfig(_Frozen):
    """One (ansatz, optimiser) pair of a benchmark.

    Attributes:
        ansatz: Trial circuit.
        optimizer: Optimiser and gradient settings.
        label: Name used in result directories; defaults to ``"<ansatz>_<optimizer label>"``.
        shots: Overrides ``backend.shots`` for this combo. Used to give every method the same
            shot budget per iteration, e.g. fewer shots per energy for parameter shift, which
            needs more energies per gradient.
    """

    ansatz: AnsatzConfig
    optimizer: OptimizerConfig = OptimizerConfig()
    label: str | None = None
    shots: Annotated[int, Field(ge=1)] | None = None
    mitigation: MitigationConfig | None = None
    mapping: MappingConfig | None = None

    def name(self) -> str:
        """``label`` or ``"<ansatz>_<optimizer label>"`` (plus the mapping if not generic)."""
        if self.label:
            return self.label
        base = f"{self.ansatz.type}_{self.optimizer.label()}"
        if self.mapping is not None and self.mapping.strategy != "generic":
            base += f"_{self.mapping.strategy}"
        return base


def _paper_combos() -> list[ComboConfig]:
    adam = OptimizerConfig(type="adam", maxiter=100, lr=0.01)
    spsa = OptimizerConfig(type="spsa", maxiter=100, lr=0.01, perturbation=0.01)
    return [
        ComboConfig(ansatz=AnsatzConfig(type="rvb"), optimizer=spsa),
        ComboConfig(ansatz=AnsatzConfig(type="rvb"), optimizer=adam),
        ComboConfig(ansatz=AnsatzConfig(type="hva"), optimizer=spsa),
        ComboConfig(ansatz=AnsatzConfig(type="hva"), optimizer=adam),
    ]


class BenchmarkConfig(_Frozen):
    """A set of runs sharing lattice and backend; defaults to the paper's four combos.

    Attributes:
        seeds: Repeat every combo once per seed. ``None`` runs each combo once with ``seed``.
            Gradient methods are stochastic, so comparisons should use several seeds; the
            summary then aggregates per combo.
        Other fields mirror :class:`RunConfig` and apply to every run.
    """

    name: str = "benchmark"
    seed: int = 42
    seeds: list[int] | None = None
    x0_scale: Annotated[float, Field(ge=0)] = math.pi / 4
    lattice: LatticeConfig = LatticeConfig()
    backend: BackendConfig = BackendConfig()
    output_dir: Path = Path("results")
    record_params: bool = False
    track_ideal_energy: bool = True
    mitigation: MitigationConfig = MitigationConfig()
    mapping: MappingConfig = MappingConfig()
    budget: BudgetConfig = BudgetConfig()
    combos: list[ComboConfig] = Field(default_factory=_paper_combos)

    def run_configs(self) -> list[RunConfig]:
        """Expand into one :class:`RunConfig` per combo and seed."""
        seeds = self.seeds or [self.seed]
        runs: list[RunConfig] = []
        for combo in self.combos:
            backend = (
                self.backend
                if combo.shots is None
                else self.backend.model_copy(update={"shots": combo.shots})
            )
            for seed in seeds:
                suffix = "" if self.seeds is None else f"-s{seed}"
                runs.append(
                    RunConfig(
                        name=f"{self.name}-{combo.name()}{suffix}",
                        seed=seed,
                        x0_scale=self.x0_scale,
                        lattice=self.lattice,
                        ansatz=combo.ansatz,
                        optimizer=combo.optimizer,
                        backend=backend,
                        output_dir=self.output_dir,
                        record_params=self.record_params,
                        track_ideal_energy=self.track_ideal_energy,
                        mitigation=combo.mitigation or self.mitigation,
                        mapping=combo.mapping or self.mapping,
                        budget=self.budget,
                    )
                )
        return runs


class Settings(BaseSettings):
    """IBM Quantum credentials from the environment (or ``.env``). Never written to disk by us.

    Attributes:
        token: API key (``IBM_QUANTUM_TOKEN``).
        instance: Cloud Resource Name of the instance (``IBM_QUANTUM_INSTANCE``); optional.
        channel: ``IBM_QUANTUM_CHANNEL``; ``"ibm_quantum_platform"`` is the only live one.
        region: Optional region hint (``IBM_QUANTUM_REGION``).
    """

    model_config = SettingsConfigDict(
        env_prefix="IBM_QUANTUM_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    token: SecretStr | None = None
    instance: str | None = None
    channel: str = "ibm_quantum_platform"
    region: str | None = None

    @property
    def has_credentials(self) -> bool:
        """Whether a token is available."""
        return self.token is not None and bool(self.token.get_secret_value())


# -- loading ----------------------------------------------------------------------------------


def load_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file into a plain dict."""
    with Path(path).open("rb") as fh:
        return tomllib.load(fh)


def parse_override(text: str) -> tuple[list[str], Any]:
    """Parse ``"a.b.c=value"`` into ``(["a", "b", "c"], value)``.

    The value is parsed as TOML (so ``1024``, ``0.1``, ``true``, ``"str"``, ``[1, 2]`` all
    work); anything that is not valid TOML is taken as a bare string.
    """
    key, sep, raw = text.partition("=")
    if not sep or not key.strip():
        msg = f"override must look like key.path=value, got {text!r}"
        raise ValueError(msg)
    try:
        value = tomllib.loads(f"v = {raw.strip()}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw.strip()
    return key.strip().split("."), value


def apply_overrides(data: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Return a copy of ``data`` with dotted ``key=value`` overrides applied."""
    result: dict[str, Any] = _deepcopy_dict(data)
    for text in overrides:
        keys, value = parse_override(text)
        node = result
        for key in keys[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                msg = f"cannot descend into non-table key {key!r} in override {text!r}"
                raise ValueError(msg)
        node[keys[-1]] = value
    return result


def load_run_config(path: Path, overrides: Iterable[str] = ()) -> RunConfig:
    """Load a :class:`RunConfig` from TOML with optional overrides."""
    return RunConfig.model_validate(apply_overrides(load_toml(path), overrides))


def load_benchmark_config(path: Path, overrides: Iterable[str] = ()) -> BenchmarkConfig:
    """Load a :class:`BenchmarkConfig` from TOML with optional overrides."""
    return BenchmarkConfig.model_validate(apply_overrides(load_toml(path), overrides))


def _deepcopy_dict(data: Mapping[str, Any]) -> dict[str, Any]:
    return {k: _deepcopy_dict(v) if isinstance(v, Mapping) else v for k, v in data.items()}


def _drop_none(data: Any) -> Any:
    """TOML has no null; strip ``None`` values recursively so defaults apply on reload."""
    if isinstance(data, dict):
        return {k: _drop_none(v) for k, v in data.items() if v is not None}
    if isinstance(data, list):
        return [_drop_none(v) for v in data]
    return data
