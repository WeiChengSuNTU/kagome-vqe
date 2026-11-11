"""Multi-seed experiment matrices for graph_hva: optimiser comparisons and depth sweeps.

A :class:`StudyConfig` (TOML, see ``configs/studies/``) describes a matrix

    lattices x reps (p) x include_h0 x backends x optimisers x seeds

and expands it into ordinary :class:`~kagome_vqe.config.RunConfig` objects. Each run goes
through :func:`kagome_vqe.results.run_and_save`, so every run is a normal results directory
(``config.toml``, ``result.json``, ``trace.csv``, ...) that ``kagome-vqe run``, ``replay``,
``plot`` and :func:`~kagome_vqe.results.load_run` understand. A study adds only a thin layer::

    results/<UTC timestamp>-<study name>/
        study.toml          the resolved study configuration
        environment.json    provenance at study start
        manifest.jsonl      one line per attempted run: labels, status, directory
        runs/<UTC timestamp>-<run name>/   one ordinary results directory per run
        summary/            runs.csv/json, groups.csv/json, report.md, figures (regenerable)

Fairness rules built in:

* every optimiser of one (lattice, p, H0, seed) starts from the same initial parameters,
  because the initial point depends only on the seed, ``x0_scale`` and the parameter count
  (:func:`kagome_vqe.vqe.initial_point`);
* resource limits are per backend (:class:`StudyBackend.budget`) and identical for every
  optimiser; they are enforced on every energy request, calibration included;
* fixed-cost optimisers (SPSA, gradient methods with the SPSA estimator) can have ``maxiter``
  derived from the budget (``fit_maxiter_to_budget``), so their gain schedules are planned
  for the evaluations they will actually get rather than cut off by the budget.

The headline parameter-quality metric is the *noise-free* energy of the final parameters
(``ideal_energy``), which for finite-shot runs separates the quality of the learned
parameters from the sampling noise of their measured energy.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import tomli_w
from pydantic import Field, field_validator, model_validator

from kagome_vqe.config import (
    AnsatzConfig,
    BackendConfig,
    BudgetConfig,
    LatticeConfig,
    OptimizerConfig,
    RunConfig,
    _Frozen,
    load_toml,
)
from kagome_vqe.hamiltonian import heisenberg_hamiltonian

__all__ = [
    "StudyBackend",
    "StudyConfig",
    "StudyLattice",
    "StudyOptimizer",
    "StudyRun",
    "estimate_study",
    "load_study_config",
    "run_study",
    "summarise_study",
]

log = logging.getLogger(__name__)

_LABEL = r"^[A-Za-z0-9_.]+$"


class StudyLattice(_Frozen):
    """One lattice of the study.

    Attributes:
        label: Short name used in run names and tables (letters, digits, ``_`` and ``.``).
        lattice: The lattice, as in a run config.
        schedule: Gate schedule for ``graph_hva`` on this lattice.
        reps: HVA cycles for this lattice, overriding the study's ``reps``.
    """

    label: Annotated[str, Field(pattern=_LABEL)]
    lattice: LatticeConfig
    schedule: Annotated[str, Field(pattern=r"^(greedy|prb105|prb105_fig3|prb106)$")] = "greedy"
    reps: list[Annotated[int, Field(ge=1)]] | None = None


class StudyBackend(_Frozen):
    """One evaluation backend with its resource budget.

    Attributes:
        label: Short name (``sv``, ``shots1024``).
        backend: Backend settings.
        budget: Limits applied identically to every optimiser on this backend.
    """

    label: Annotated[str, Field(pattern=_LABEL)]
    backend: BackendConfig
    budget: BudgetConfig = BudgetConfig()


class StudyOptimizer(_Frozen):
    """One optimiser configuration.

    Attributes:
        label: Short name (``lbfgsb``, ``spsa_cal``, ``adam_spsa``).
        optimizer: Optimiser settings.
        backends: Restrict to these backend labels (``None``: all backends).
        fit_maxiter_to_budget: For optimisers with a fixed cost per iteration, set ``maxiter``
            so that calibration, iterations and the final evaluation exactly use the
            backend's budget (see :func:`fitted_maxiter`). Ignored without a budget.
    """

    label: Annotated[str, Field(pattern=_LABEL)]
    optimizer: OptimizerConfig
    backends: list[str] | None = None
    fit_maxiter_to_budget: bool = False


class StudyRun(_Frozen):
    """One cell of the expanded matrix: its labels and run configuration."""

    lattice: str
    reps: int
    include_h0: bool
    backend: str
    optimizer: str
    seed: int
    config: RunConfig

    @property
    def group(self) -> str:
        """Everything except the seed: runs that should be aggregated together."""
        h0 = "h0" if self.include_h0 else "noh0"
        return f"{self.lattice}-p{self.reps}-{h0}-{self.backend}-{self.optimizer}"

    def labels(self) -> dict[str, Any]:
        """The cell's labels as a dictionary."""
        return {
            "lattice": self.lattice,
            "reps": self.reps,
            "include_h0": self.include_h0,
            "backend": self.backend,
            "optimizer": self.optimizer,
            "seed": self.seed,
            "group": self.group,
            "name": self.config.name,
        }


class StudyConfig(_Frozen):
    """An experiment matrix (see the module docstring).

    Attributes:
        name: Study name (results directory suffix and run-name prefix).
        description: Free text copied into the report.
        output_dir: Parent of the study directory.
        seeds: Independent repetitions; each seed fixes the initial point and all randomness.
        x0_scale: Initial parameters uniform in ``[0, x0_scale)``.
        reps: HVA cycles ``p`` to scan.
        include_h0: H0 settings to scan.
        ansatz: Template for the remaining ansatz options (type must be ``graph_hva``).
        lattices: Lattices.
        backends: Backends with budgets.
        optimizers: Optimisers.
        success_threshold: A run succeeds if the noise-free relative energy error of its
            final parameters is below this.
        record_params: Save every evaluated parameter vector (``params.npy``).
        track_ideal_energy: Record the noise-free energy after every iteration (needed for
            the error-versus-cost figures).
    """

    name: Annotated[str, Field(pattern=_LABEL.replace("_.", "_.-"))] = "study"
    description: str = ""
    output_dir: Path = Path("results")
    seeds: Annotated[list[int], Field(min_length=1)] = Field(default_factory=lambda: [1])
    x0_scale: Annotated[float, Field(ge=0)] = 0.1
    reps: Annotated[list[Annotated[int, Field(ge=1)]], Field(min_length=1)] = Field(
        default_factory=lambda: [1]
    )
    include_h0: Annotated[list[bool], Field(min_length=1)] = Field(default_factory=lambda: [False])
    ansatz: AnsatzConfig = AnsatzConfig(type="graph_hva")
    lattices: Annotated[list[StudyLattice], Field(min_length=1)]
    backends: Annotated[list[StudyBackend], Field(min_length=1)]
    optimizers: Annotated[list[StudyOptimizer], Field(min_length=1)]
    success_threshold: Annotated[float, Field(gt=0)] = 0.01
    record_params: bool = True
    track_ideal_energy: bool = True

    @field_validator("ansatz")
    @classmethod
    def _graph_only(cls, value: AnsatzConfig) -> AnsatzConfig:
        if value.type != "graph_hva":
            msg = "studies scan graph_hva options (reps, include_h0, schedule); type must match"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _labels_unique_and_known(self) -> StudyConfig:
        for kind, items in (
            ("lattice", self.lattices),
            ("backend", self.backends),
            ("optimizer", self.optimizers),
        ):
            labels = [item.label for item in items]
            if len(set(labels)) != len(labels):
                msg = f"duplicate {kind} labels: {labels}"
                raise ValueError(msg)
        known = {b.label for b in self.backends}
        for opt in self.optimizers:
            unknown = set(opt.backends or ()) - known
            if unknown:
                msg = f"optimizer {opt.label!r} names unknown backends {sorted(unknown)}"
                raise ValueError(msg)
        if len(set(self.seeds)) != len(self.seeds):
            msg = "seeds must be distinct"
            raise ValueError(msg)
        return self

    def runs(self) -> list[StudyRun]:
        """Expand the matrix.

        Order: lattice, p, H0, backend, seed, then optimiser, so that an interrupted study
        has complete optimiser comparisons for the seeds it finished.
        """
        out: list[StudyRun] = []
        for lat in self.lattices:
            for reps in lat.reps or self.reps:
                for h0 in self.include_h0:
                    ansatz = self.ansatz.model_copy(
                        update={"reps": reps, "include_h0": h0, "schedule": lat.schedule}
                    )
                    num_parameters = ansatz.build(lat.lattice.build()).num_parameters
                    for back in self.backends:
                        for seed in self.seeds:
                            for opt in self.optimizers:
                                if opt.backends is not None and back.label not in opt.backends:
                                    continue
                                optimizer = opt.optimizer
                                if opt.fit_maxiter_to_budget:
                                    fitted = fitted_maxiter(
                                        optimizer, back, lat.lattice, num_parameters
                                    )
                                    if fitted is not None:
                                        optimizer = optimizer.model_copy(update={"maxiter": fitted})
                                h0_tag = "h0" if h0 else "noh0"
                                name = (
                                    f"{self.name}-{lat.label}-p{reps}-{h0_tag}-{back.label}-"
                                    f"{opt.label}-s{seed}"
                                )
                                cfg = RunConfig(
                                    name=name,
                                    seed=seed,
                                    x0_scale=self.x0_scale,
                                    lattice=lat.lattice,
                                    ansatz=ansatz,
                                    optimizer=optimizer,
                                    backend=back.backend,
                                    budget=back.budget,
                                    record_params=self.record_params,
                                    track_ideal_energy=self.track_ideal_energy,
                                )
                                out.append(
                                    StudyRun(
                                        lattice=lat.label,
                                        reps=reps,
                                        include_h0=h0,
                                        backend=back.label,
                                        optimizer=opt.label,
                                        seed=seed,
                                        config=cfg,
                                    )
                                )
        return out

    def to_toml_dict(self) -> dict[str, Any]:
        """TOML-safe dictionary."""
        data = self.model_dump(mode="json", exclude_none=True)
        return dict(data)


def load_study_config(path: Path) -> StudyConfig:
    """Read a study TOML."""
    return StudyConfig.model_validate(load_toml(Path(path)))


def evaluation_budget(backend: StudyBackend, lattice: LatticeConfig) -> int | None:
    """Energy evaluations the backend's budget allows (the tighter of its two limits).

    A shot limit converts at ``shots * measurement_groups`` per evaluation (no mitigation in
    studies), with the groups counted on the lattice's Hamiltonian.
    """
    limits: list[int] = []
    if backend.budget.max_evaluations is not None:
        limits.append(backend.budget.max_evaluations)
    if backend.budget.max_shots is not None and backend.backend.shots:
        hamiltonian = heisenberg_hamiltonian(lattice.build())
        groups = len(hamiltonian.group_commuting(qubit_wise=True))
        limits.append(backend.budget.max_shots // (backend.backend.shots * groups))
    return min(limits) if limits else None


def fitted_maxiter(
    optimizer: OptimizerConfig,
    backend: StudyBackend,
    lattice: LatticeConfig,
    num_parameters: int,
) -> int | None:
    """``maxiter`` that spends exactly the evaluation budget, for fixed-cost optimisers.

    * calibrated or plain SPSA: ``(E - calibration - 1) // 2`` (two per iteration, one final);
    * ADAM / GD: ``(E - 1) // k`` with ``k`` the gradient's energies per iteration
      (2 for SPSA, ``n + 1`` forward, ``2n + 1`` central);
    * scipy: ``None`` (variable cost; the budget guard stops it).
    """
    budget = evaluation_budget(backend, lattice)
    if budget is None:
        return None
    if optimizer.type == "spsa":
        calibration = 2 * optimizer.calibration_samples if optimizer.calibrate else 0
        iterations = (budget - calibration - 1) // 2
    elif optimizer.type in ("adam", "gd"):
        per_iteration = {
            "spsa": 2,
            "forward": num_parameters + 1,
            "central": 2 * num_parameters + 1,
        }.get(optimizer.gradient)
        if per_iteration is None:
            return None
        iterations = (budget - 1) // per_iteration
    else:
        return None
    if iterations < 1:
        msg = f"budget of {budget} evaluations leaves no iteration for {optimizer.label()}"
        raise ValueError(msg)
    return int(iterations)


# -- running ----------------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _read_manifest(study_dir: Path) -> list[dict[str, Any]]:
    path = study_dir / "manifest.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append_manifest(study_dir: Path, row: dict[str, Any]) -> None:
    with (study_dir / "manifest.jsonl").open("a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def new_study_dir(cfg: StudyConfig, root: Path | None = None) -> Path:
    """Create ``<root>/<UTC timestamp>-<name>/`` (never reusing an existing directory)."""
    root = Path(root or cfg.output_dir)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = root / f"{stamp}-{cfg.name}"
    counter = 1
    while path.exists():
        counter += 1
        path = root / f"{stamp}-{cfg.name}-{counter}"
    path.mkdir(parents=True)
    return path


def run_study(
    cfg: StudyConfig,
    *,
    study_dir: Path | None = None,
    root: Path | None = None,
    only: Iterable[str] | None = None,
    max_runs: int | None = None,
    time_limit_s: float | None = None,
    on_result: Callable[[StudyRun, Any, Path], None] | None = None,
    summarise: bool = True,
) -> Path:
    """Execute the study's runs that have not completed yet, then write the summary.

    Args:
        cfg: The study.
        study_dir: Resume this existing study directory; its completed runs (manifest status
            ``completed``) are skipped. A failed or interrupted run is retried into a *new*
            run directory; the old one is kept.
        root: Parent for a new study directory (default ``cfg.output_dir``).
        only: Run only cells whose run name contains one of these substrings.
        max_runs: Stop after starting this many runs (the rest can be resumed).
        time_limit_s: Do not start a new run after this many seconds.
        on_result: Called with the cell, the :class:`~kagome_vqe.vqe.VQERunResult` and the run
            directory after each completed run.
        summarise: Write ``summary/`` at the end.

    Returns:
        The study directory.
    """
    from kagome_vqe.results import RunFailedError, environment_info, run_and_save

    if study_dir is None:
        study_dir = new_study_dir(cfg, root)
        with (study_dir / "study.toml").open("wb") as fh:
            tomli_w.dump(cfg.to_toml_dict(), fh)
        (study_dir / "environment.json").write_text(
            json.dumps(environment_info(), indent=2, default=str) + "\n"
        )
    else:
        study_dir = Path(study_dir)
        saved = load_study_config(study_dir / "study.toml")
        if saved != cfg:
            msg = f"{study_dir}/study.toml differs from the given configuration; not resuming"
            raise ValueError(msg)
    done = {row["name"] for row in _read_manifest(study_dir) if row["status"] == "completed"}
    filters = list(only or ())
    cells = [
        cell
        for cell in cfg.runs()
        if cell.config.name not in done
        and (not filters or any(f in cell.config.name for f in filters))
    ]
    log.info("study %s: %d runs to do in %s", cfg.name, len(cells), study_dir)
    start = time.perf_counter()
    for k, cell in enumerate(cells):
        if max_runs is not None and k >= max_runs:
            log.info("max_runs=%d reached; resume with --resume %s", max_runs, study_dir)
            break
        if time_limit_s is not None and time.perf_counter() - start > time_limit_s:
            log.info("time limit reached; resume with --resume %s", study_dir)
            break
        log.info("[%d/%d] %s", k + 1, len(cells), cell.config.name)
        row = {**cell.labels(), "started_utc": _now()}

        def forward(result: Any, path: Path, cell: StudyRun = cell) -> None:
            if on_result is not None:
                on_result(cell, result, path)

        try:
            run_dir = run_and_save(cell.config, root=study_dir / "runs", on_result=forward)
        except RunFailedError as exc:
            _append_manifest(
                study_dir,
                {**row, "status": "failed", "error": str(exc), "path": str(exc.run_dir)},
            )
            continue
        except KeyboardInterrupt:
            _append_manifest(study_dir, {**row, "status": "interrupted"})
            raise
        _append_manifest(
            study_dir,
            {**row, "status": "completed", "path": str(run_dir.relative_to(study_dir))},
        )
    if summarise:
        summarise_study(study_dir)
    return study_dir


def estimate_study(
    cfg: StudyConfig, *, probe_evaluations: int = 5
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Plan and cost estimate without optimising: runs, evaluation budgets, time per run.

    Time per evaluation is measured by evaluating ``probe_evaluations`` random points once
    per (lattice, p, H0, backend); the run time estimate is that times the budget (or, without
    a budget, ``maxiter`` times the cost per iteration where it is known). Real runs also
    spend time on noise-free tracking and stop early when an optimiser converges.
    """
    from kagome_vqe.estimators import make_estimator
    from kagome_vqe.vqe import prepare

    rows: list[dict[str, Any]] = []
    per_eval: dict[tuple[str, int, bool, str], float] = {}
    for cell in cfg.runs():
        key = (cell.lattice, cell.reps, cell.include_h0, cell.backend)
        prep = prepare(cell.config)
        if key not in per_eval:
            rng = np.random.default_rng(0)
            points = rng.random((probe_evaluations, prep.parameter_map.num_occurrences))
            with make_estimator(
                cell.config.backend, prep.problem, prep.backend, seed=0
            ) as estimator:
                estimator.evaluate(points[:1])  # warm-up (compilation caches)
                tick = time.perf_counter()
                for point in points:
                    estimator.evaluate(point[None, :])
                per_eval[key] = (time.perf_counter() - tick) / probe_evaluations
        back = next(b for b in cfg.backends if b.label == cell.backend)
        budget = evaluation_budget(back, cell.config.lattice)
        n = prep.ansatz.num_parameters
        if budget is None and prep.projected_evaluations is not None:
            budget = prep.projected_evaluations
        groups = prep.problem.measurement_groups
        shots = cell.config.backend.shots
        rows.append(
            {
                **cell.labels(),
                "num_parameters": n,
                "maxiter": cell.config.optimizer.maxiter,
                "evaluation_budget": budget,
                "shots_budget": None
                if budget is None or cell.config.backend.is_exact or not shots
                else budget * shots * groups,
                "seconds_per_evaluation": per_eval[key],
                "estimated_seconds": None if budget is None else budget * per_eval[key],
            }
        )
    frame = pd.DataFrame(rows)
    known = frame["estimated_seconds"].dropna()
    totals = {
        "runs": len(frame),
        "runs_without_estimate": int(frame["estimated_seconds"].isna().sum()),
        "estimated_hours": float(known.sum() / 3600),
        "evaluations": int(frame["evaluation_budget"].fillna(0).sum()),
        "shots": int(frame["shots_budget"].fillna(0).sum()),
    }
    return frame, totals


# -- summaries --------------------------------------------------------------------------------


def _relative(exact: float, energy: float | None) -> float:
    if energy is None or not math.isfinite(energy):
        return math.nan
    return abs(energy - exact) / abs(exact)


def _run_row(labels: dict[str, Any], run: Any, threshold: float) -> dict[str, Any]:
    res = run.result
    resources = run.resources
    exact = float(res["exact_energy"])
    ideal = res.get("ideal_energy")
    if ideal is None and run.config.backend.is_exact:
        ideal = res["final_energy"]
    ideal_err = _relative(exact, ideal)
    return {
        **{k: labels[k] for k in ("lattice", "reps", "include_h0", "backend", "optimizer")},
        "seed": int(res.get("seed", labels["seed"])),
        "group": labels["group"],
        "num_parameters": res.get("num_parameters", len(res["optimal_params"])),
        "exact_energy": exact,
        "final_energy": res["final_energy"],
        "final_relative_error": _relative(exact, res["final_energy"]),
        "ideal_energy": ideal,
        "ideal_relative_error": ideal_err,
        "fidelity": res.get("ideal_fidelity"),
        "evaluations": res["nfev"],
        "calibration_evaluations": resources.get("calibration_evaluations", 0),
        "iterations": res["nit"],
        "total_shots": res.get("total_shots", 0),
        "optimiser_wall_time_s": resources.get("optimiser_wall_time_s", res["wall_time_s"]),
        "wall_time_s": res["wall_time_s"],
        "termination": run.metadata.get("termination", {}).get("reason", "optimizer"),
        "message": res.get("optimizer_message", ""),
        "gradient": run.metadata.get("optimizer", {}).get("gradient"),
        "success": bool(ideal_err < threshold),
        "path": str(run.path),
    }


def collect_runs(study_dir: Path, *, threshold: float = 0.01) -> pd.DataFrame:
    """One row per completed run of a study (from the manifest and the saved results)."""
    from kagome_vqe.results import load_run

    study_dir = Path(study_dir)
    rows = []
    latest: dict[str, dict[str, Any]] = {}
    for row in _read_manifest(study_dir):
        if row["status"] == "completed":
            latest[row["name"]] = row
    for labels in latest.values():
        path = Path(labels["path"])
        run = load_run(path if path.is_absolute() else study_dir / path)
        rows.append(_run_row(labels, run, threshold))
    return pd.DataFrame(rows)


def _group_stats(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["lattice", "reps", "include_h0", "backend", "optimizer"]
    records = []
    for key, group in frame.groupby(keys, sort=True):
        err = group["ideal_relative_error"].astype(float)
        energy = group["ideal_energy"].astype(float)
        fidelity = pd.to_numeric(group["fidelity"], errors="coerce")
        records.append(
            {
                **dict(zip(keys, key, strict=True)),
                "seeds": len(group),
                "num_parameters": int(group["num_parameters"].iloc[0]),
                "mean_ideal_energy": energy.mean(),
                "std_ideal_energy": energy.std(ddof=1) if len(group) > 1 else 0.0,
                "median_relative_error": err.median(),
                "best_relative_error": err.min(),
                "worst_relative_error": err.max(),
                "mean_fidelity": fidelity.mean(),
                "std_fidelity": fidelity.std(ddof=1) if len(group) > 1 else 0.0,
                "mean_evaluations": group["evaluations"].mean(),
                "mean_total_shots": group["total_shots"].mean(),
                "mean_optimiser_time_s": group["optimiser_wall_time_s"].mean(),
                "successes": int(group["success"].sum()),
                "success_rate": group["success"].mean(),
                "budget_stops": int((group["termination"] == "budget").sum()),
            }
        )
    return pd.DataFrame(records)


def _fmt(value: Any, spec: str) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return format(value, spec)


def _markdown_report(
    cfg: StudyConfig | None, runs: pd.DataFrame, groups: pd.DataFrame, figures: list[Path]
) -> str:
    lines = [f"# Study report: {cfg.name if cfg else 'study'}", ""]
    lines += [
        "Generated by `kagome-vqe study --summarise` from the saved runs in `runs/`; every",
        "number below is read from those result files. Statistics are over seeds; they",
        "describe these seeds on this lattice only and do not establish a general ranking.",
        "",
    ]
    if cfg is not None:
        if cfg.description:
            lines += [cfg.description.strip(), ""]
        lines += [
            "## Configuration",
            "",
            "Hamiltonian: H = sum over bonds of J (XX + YY + ZZ), Pauli normalisation.",
            f"Seeds: {cfg.seeds}; initial parameters uniform in [0, {cfg.x0_scale}); "
            "the same initial point for every optimiser of a seed.",
            f"Success: noise-free relative energy error of the final parameters below "
            f"{cfg.success_threshold:g}.",
            "",
            "| lattice | definition | schedule |",
            "|---|---|---|",
        ]
        lines += [
            f"| {lat.label} | `{lat.lattice.model_dump(mode='json', exclude_none=True)}` "
            f"| {lat.schedule} |"
            for lat in cfg.lattices
        ]
        lines += ["", "| backend | settings | budget |", "|---|---|---|"]
        lines += [
            f"| {b.label} | {b.backend.label()}"
            f"{'' if b.backend.is_exact else f', {b.backend.shots} shots per basis'} | "
            f"{b.budget.model_dump(exclude_none=True) or 'none'} |"
            for b in cfg.backends
        ]
        lines += ["", "| optimizer | settings | gradient actually used |", "|---|---|---|"]
        lines += [
            f"| {o.label} | `{o.optimizer.model_dump(mode='json', exclude_defaults=True)}`"
            f"{' (maxiter fitted to budget)' if o.fit_maxiter_to_budget else ''} | "
            f"{o.optimizer.gradient_description()} |"
            for o in cfg.optimizers
        ]
        lines.append("")
    if groups.empty:
        return "\n".join([*lines, "No completed runs yet.", ""])
    lines += [
        "## Aggregated over seeds",
        "",
        "Energies and errors are noise-free values of the final parameters "
        "(for shot-based backends: the statevector energy at the parameters the optimiser "
        "returned). Evaluations include calibration; shots include the final re-measurement.",
        "",
        "| lattice | p | H0 | backend | optimizer | n_theta | seeds | mean E (+- std) "
        "| median rel. err | best / worst rel. err | mean fidelity | mean evals "
        "| mean shots | mean opt. time (s) | success | budget stops |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for g in groups.to_dict("records"):
        lines.append(
            f"| {g['lattice']} | {g['reps']} | {'yes' if g['include_h0'] else 'no'} | "
            f"{g['backend']} | {g['optimizer']} | {g['num_parameters']} | {g['seeds']} | "
            f"{_fmt(g['mean_ideal_energy'], '.5f')} +- {_fmt(g['std_ideal_energy'], '.5f')} | "
            f"{_fmt(g['median_relative_error'], '.2e')} | "
            f"{_fmt(g['best_relative_error'], '.2e')} / {_fmt(g['worst_relative_error'], '.2e')}"
            f" | {_fmt(g['mean_fidelity'], '.4f')} | {_fmt(g['mean_evaluations'], '.0f')} | "
            f"{_fmt(g['mean_total_shots'], ',.0f')} | {_fmt(g['mean_optimiser_time_s'], '.1f')} | "
            f"{g['successes']}/{g['seeds']} | {g['budget_stops']} |"
        )
    lines += [
        "",
        "## Per run",
        "",
        "| lattice | p | H0 | backend | optimizer | seed | ideal rel. err | measured final E "
        "| fidelity | evals (calib.) | iterations | shots | opt. time (s) | stop |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs.sort_values(
        ["lattice", "reps", "include_h0", "backend", "optimizer", "seed"]
    ).to_dict("records"):
        lines.append(
            f"| {r['lattice']} | {r['reps']} | {'yes' if r['include_h0'] else 'no'} | "
            f"{r['backend']} | {r['optimizer']} | {r['seed']} | "
            f"{_fmt(r['ideal_relative_error'], '.2e')} | {_fmt(r['final_energy'], '.5f')} | "
            f"{_fmt(r['fidelity'], '.4f')} | {r['evaluations']} ({r['calibration_evaluations']})"
            f" | {r['iterations']} | {_fmt(r['total_shots'], ',')} | "
            f"{_fmt(r['optimiser_wall_time_s'], '.1f')} | {r['termination']} |"
        )
    if figures:
        lines += ["", "## Figures", ""]
        lines += [f"![{fig.stem}](figures/{fig.name})" for fig in figures]
    lines.append("")
    return "\n".join(lines)


def summarise_study(study_dir: Path) -> Path:
    """(Re)write ``summary/`` of a study from its saved runs; returns the report path."""
    from kagome_vqe.plotting import plot_study_figures

    study_dir = Path(study_dir)
    cfg_path = study_dir / "study.toml"
    cfg = load_study_config(cfg_path) if cfg_path.exists() else None
    threshold = cfg.success_threshold if cfg else 0.01
    runs = collect_runs(study_dir, threshold=threshold)
    out = study_dir / "summary"
    out.mkdir(exist_ok=True)
    groups = _group_stats(runs) if not runs.empty else pd.DataFrame()
    runs.to_csv(out / "runs.csv", index=False)
    (out / "runs.json").write_text(runs.to_json(orient="records", indent=2) + "\n")
    groups.to_csv(out / "groups.csv", index=False)
    (out / "groups.json").write_text(groups.to_json(orient="records", indent=2) + "\n")
    figures = [] if runs.empty else plot_study_figures(study_dir, runs, out / "figures")
    report = out / "report.md"
    report.write_text(_markdown_report(cfg, runs, groups, figures))
    return report
