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

from kagome_vqe.ansatz import Ansatz, HVAnsatz, LibraryAnsatz, RVBAnsatz
from kagome_vqe.ansatz.base import ring_bonds
from kagome_vqe.lattice import KAGOME_HAMILTONIAN_CYCLE, Lattice, kagome_unit_cell
from kagome_vqe.optimizers import Adam, Optimizer, ScipyOptimizer, Spsa

__all__ = [
    "AnsatzConfig",
    "BackendConfig",
    "BackendKind",
    "BenchmarkConfig",
    "ComboConfig",
    "LatticeConfig",
    "OptimizerConfig",
    "RunConfig",
    "Settings",
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


class LatticeConfig(_Frozen):
    """Which spin graph to solve.

    Attributes:
        kind: Only the 12-site kagome unit cell is shipped; the field exists so larger lattices
            can be added without changing config files.
        t: Exchange coupling on every bond.
        site_order: How ansatz qubits map to lattice sites. ``"hamiltonian_cycle"`` (default)
            makes the ring ansatzes exact; ``"identity"`` is the deliberately hard variant;
            or an explicit permutation.
    """

    kind: Literal["kagome_unit_cell"] = "kagome_unit_cell"
    t: float = 1.0
    site_order: Literal["hamiltonian_cycle", "identity"] | list[int] = "hamiltonian_cycle"

    def build(self) -> Lattice:
        """Construct the lattice."""
        if self.site_order == "hamiltonian_cycle":
            order: tuple[int, ...] | None = KAGOME_HAMILTONIAN_CYCLE
        elif self.site_order == "identity":
            order = None
        else:
            order = tuple(self.site_order)
        return kagome_unit_cell(self.t, site_order=order)


class AnsatzConfig(_Frozen):
    """Which trial circuit to use.

    Attributes:
        type: ``"rvb"``, ``"hva"``, ``"real_amplitudes"`` or ``"efficient_su2"``.
        reps: Number of layers.
        gate: RVB only: ``"pauli_evolution"`` (true eSWAP) or ``"legacy"`` (original gate).
        lattice_aware: HVA only: add a third sub-layer on the lattice bonds that are not on
            the ansatz ring.
        entanglement: Library ansatzes only: Qiskit entanglement pattern.
    """

    type: Literal["rvb", "hva", "real_amplitudes", "efficient_su2"] = "hva"
    reps: Annotated[int, Field(ge=1)] = 1
    gate: Literal["pauli_evolution", "legacy"] = "pauli_evolution"
    lattice_aware: bool = False
    entanglement: str = "reverse_linear"

    def build(self, lattice: Lattice) -> Ansatz:
        """Construct the ansatz for ``lattice.num_sites`` qubits."""
        n = lattice.num_sites
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


class OptimizerConfig(_Frozen):
    """Classical optimiser settings.

    Attributes:
        type: ``"adam"``, ``"spsa"`` or ``"scipy"``.
        maxiter: Iteration budget.
        lr: ADAM learning rate / SPSA step size.
        perturbation: SPSA perturbation magnitude.
        fd_eps: ADAM finite-difference step (use ~0.1 under shot noise).
        gradient: ADAM difference scheme.
        calibrate: SPSA: derive decaying gains automatically instead of fixed ``lr``.
        method: scipy method name.
        tol: Convergence tolerance (ADAM update norm / scipy ``tol``).
    """

    type: Literal["adam", "spsa", "scipy"] = "adam"
    maxiter: Annotated[int, Field(ge=1)] = 100
    lr: Annotated[float, Field(gt=0)] = 0.01
    perturbation: Annotated[float, Field(gt=0)] = 0.01
    fd_eps: Annotated[float, Field(gt=0)] = 1e-6
    gradient: Literal["forward", "central"] = "forward"
    calibrate: bool = False
    method: str = "COBYLA"
    tol: float | None = None

    def build(self, rng: np.random.Generator) -> Optimizer:
        """Construct the optimiser; ``rng`` seeds SPSA's random directions."""
        if self.type == "adam":
            return Adam(
                maxiter=self.maxiter,
                lr=self.lr,
                fd_eps=self.fd_eps,
                gradient=self.gradient,
                tol=1e-6 if self.tol is None else self.tol,
            )
        if self.type == "spsa":
            return Spsa(
                maxiter=self.maxiter,
                learning_rate=self.lr,
                perturbation=self.perturbation,
                calibrate=self.calibrate,
                seed=rng,
            )
        return ScipyOptimizer(method=self.method, maxiter=self.maxiter, tol=self.tol)


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

    @model_validator(mode="after")
    def _needs_name_and_shots(self) -> Self:
        if self.kind in ("aer_noisy", "fake", "ibm") and not self.name:
            msg = f"backend.kind={self.kind!r} requires backend.name"
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
        return self.kind if self.name is None else f"{self.kind}:{self.name}"


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
    """One (ansatz, optimiser) pair of a benchmark."""

    ansatz: AnsatzConfig
    optimizer: OptimizerConfig
    label: str | None = None

    def name(self) -> str:
        """``label`` or ``"<ansatz>_<optimizer>"``."""
        return self.label or f"{self.ansatz.type}_{self.optimizer.type}"


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
    """A set of runs sharing lattice, backend and seed; defaults to the paper's four combos."""

    name: str = "benchmark"
    seed: int = 42
    x0_scale: Annotated[float, Field(ge=0)] = math.pi / 4
    lattice: LatticeConfig = LatticeConfig()
    backend: BackendConfig = BackendConfig()
    output_dir: Path = Path("results")
    record_params: bool = False
    combos: list[ComboConfig] = Field(default_factory=_paper_combos)

    def run_configs(self) -> list[RunConfig]:
        """Expand into one :class:`RunConfig` per combo."""
        return [
            RunConfig(
                name=f"{self.name}-{combo.name()}",
                seed=self.seed,
                x0_scale=self.x0_scale,
                lattice=self.lattice,
                ansatz=combo.ansatz,
                optimizer=combo.optimizer,
                backend=self.backend,
                output_dir=self.output_dir,
                record_params=self.record_params,
            )
            for combo in self.combos
        ]


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
