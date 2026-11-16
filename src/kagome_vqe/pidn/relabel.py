"""Re-measure the Stage I targets of a recorded trajectory through another mitigation pipeline.

Question this answers: *does a better error-mitigation pipeline give PIDN training data it can
learn from?* To compare pipelines on equal terms, the parameter path is held fixed: the points
``theta_0 .. theta_{T-1}`` of a recorded ZNE gradient-descent run are re-measured -- cost and
parameter-shift gradient (``2K + 1`` energies per point) -- through the pipeline (noise model,
shots, readout correction, symmetry filter, ZNE, twin rescaling). At every recorded point,
including the later ones a surrogate is tested on, the raw noisy cost (PIDN's input) is also
measured once.

The output is an ordinary run directory per (pipeline, source) with ``config.toml`` (the
source's configuration with the pipeline's backend and mitigation), ``result.json``,
``trace.csv`` and a ``trajectory.npz`` of the gradient-descent schema, so the PIDN workflow
(:mod:`kagome_vqe.pidn.workflow`) and the open-loop test (:mod:`kagome_vqe.pidn.diagnostics`)
accept it as a source. Rows ``t >= t_init`` carry only the raw cost and the noise-free
reference (their gradient and value are NaN). Cost columns count the pipeline's gradient
batches; the raw costs are bookkeeping (in an experiment they come from the same shots: scale 1
of ZNE, or the unfiltered histograms).
"""

from __future__ import annotations

import glob
import json
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import tomli_w
from pydantic import BaseModel, ConfigDict, Field

from kagome_vqe.config import BackendConfig, MitigationConfig, RunConfig, load_toml

__all__ = ["RelabelConfig", "load_relabel_config", "relabel_source", "run_relabel"]

log = logging.getLogger(__name__)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RelabelPipeline(_Frozen):
    """A measurement pipeline: backend (noise model, shots) and mitigation stack."""

    label: Annotated[str, Field(pattern=r"^[A-Za-z0-9_.]+$")]
    backend: BackendConfig
    mitigation: MitigationConfig = MitigationConfig()


class RelabelConfig(_Frozen):
    """Pipelines x source runs (TOML in ``configs/pidn/``).

    Attributes:
        name: Output directory suffix.
        description: Free text.
        sources: Gradient-descent run directories (glob patterns) with ``trajectory.npz``.
        root: Base for relative patterns.
        t_init: Points whose cost and gradient are re-measured.
        seed_offset: Added to the source seed for the pipeline's randomness.
        pipelines: The pipelines.
    """

    name: str = "graph_hva_pidn_relabel"
    description: str = ""
    sources: list[str]
    root: Path | None = None
    output_dir: Path = Path("results")
    t_init: Annotated[int, Field(ge=2)] = 10
    seed_offset: int = 104_729
    pipelines: Annotated[list[RelabelPipeline], Field(min_length=1)]


def load_relabel_config(path: Path) -> RelabelConfig:
    """Read a relabel TOML."""
    return RelabelConfig.model_validate(load_toml(Path(path)))


def _sources(cfg: RelabelConfig) -> list[Path]:
    base = Path.cwd() if cfg.root is None else Path(cfg.root)
    found: list[Path] = []
    for pattern in cfg.sources:
        full = pattern if Path(pattern).is_absolute() else str(base / pattern)
        found.extend(
            Path(m) for m in sorted(glob.glob(full)) if (Path(m) / "trajectory.npz").exists()
        )
    if not found:
        msg = f"no runs with trajectory.npz match {cfg.sources}"
        raise FileNotFoundError(msg)
    return found


def relabel_source(
    pipeline: RelabelPipeline, source: Path, out_dir: Path, *, t_init: int, seed_offset: int
) -> dict[str, Any]:
    """Re-measure ``source``'s first ``t_init`` points through ``pipeline`` into ``out_dir``."""
    from kagome_vqe.estimators import make_estimator
    from kagome_vqe.optimizers import ParameterShift
    from kagome_vqe.results import load_run
    from kagome_vqe.vqe import EnergyObjective, prepare

    tick = time.perf_counter()
    run = load_run(source)
    traj = run.trajectory
    assert traj is not None
    cfg = _relabelled_config(run.config, pipeline, seed_offset)
    prep = prepare(cfg)
    mapping = prep.parameter_map
    groups = prep.problem.measurement_groups
    zne = cfg.mitigation.zne
    scales = len(zne.scales) if zne is not None else 1
    shots = int(cfg.backend.shots or 0)
    rows = len(traj["value"])
    n = traj["theta"].shape[1]
    value = np.full(rows, np.nan)
    value_std = np.full(rows, np.nan)
    grad = np.full((rows, n), np.nan)
    raw = np.full(rows, np.nan)
    evaluations = np.zeros(rows, dtype=np.int64)
    with (
        make_estimator(
            cfg.backend,
            prep.problem,
            None,
            seed=cfg.seed,
            mitigation=cfg.mitigation,
            twin=prep.circuit_twin,
        ) as estimator,
        make_estimator(cfg.backend, prep.problem, None, seed=cfg.seed + 1) as raw_estimator,
    ):
        objective = EnergyObjective(estimator, parameter_map=mapping)
        shift = ParameterShift(mapping, include_value=True)
        total = 0
        for t in range(rows):
            theta = traj["theta"][t]
            raw[t] = float(raw_estimator.evaluate(mapping.expand(theta)[None, :]).energies[0])
            if t < t_init:
                first = len(objective.trace)
                est = shift.estimate(objective, theta)
                grad[t] = est.grad
                value[t] = float(np.nan if est.value is None else est.value)
                value_std[t] = objective.trace[first].std
                total += est.nfev
            evaluations[t] = total
        description = estimator.description
    trajectory = {
        "theta": traj["theta"],
        "theta_next": traj["theta_next"],
        "grad": grad,
        "value": value,
        "value_std": value_std,
        "scales": np.ones(1),
        "scale_values": raw[:, None],
        "scale_stds": np.full((rows, 1), np.nan),
        "scale_grads": grad[:, None, :],
        "zne_weights": np.ones((rows, 1)),
        "learning_rate": traj["learning_rate"],
        "evaluations": evaluations,
        "circuit_executions": evaluations * groups * scales,
        "shots": evaluations * groups * scales * shots,
        "ideal_energy": traj["ideal_energy"],
        "ideal_grad": traj["ideal_grad"],
    }
    _write_files(out_dir, cfg, trajectory, objective.trace)
    seconds = time.perf_counter() - tick
    cos = _cosines(grad[:t_init], traj["ideal_grad"][:t_init])
    result = {
        "schema": 2,
        "name": cfg.name,
        "seed": cfg.seed,
        "relabelled_from": str(source),
        "pipeline": pipeline.label,
        "exact_energy": float(run.result["exact_energy"]),
        "final_energy": math.nan,
        "ideal_energy": None,
        "optimal_params": [float(x) for x in traj["theta_next"][t_init - 1]],
        "t_init": t_init,
        "estimator": description,
        "wall_time_s": seconds,
        "gradient_cosine_to_ideal": [float(c) for c in cos],
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    (out_dir / "status.json").write_text(json.dumps({"status": "completed"}) + "\n")
    log.info(
        "%s: mean cos(g, g_ideal) %.3f over %d points (%.0f s)",
        out_dir.name,
        float(np.nanmean(cos)),
        t_init,
        seconds,
    )
    return result


def _write_files(
    out_dir: Path, cfg: RunConfig, trajectory: dict[str, np.ndarray], trace: list[Any]
) -> None:
    from kagome_vqe.results import environment_info

    out_dir.mkdir(parents=True, exist_ok=False)
    with (out_dir / "config.toml").open("wb") as fh:
        tomli_w.dump(cfg.to_toml_dict(), fh)
    np.savez(out_dir / "trajectory.npz", **trajectory)  # type: ignore[arg-type]
    pd.DataFrame([asdict(r) for r in trace]).to_csv(out_dir / "trace.csv", index=False)
    (out_dir / "environment.json").write_text(json.dumps(environment_info(), indent=2) + "\n")


def _relabelled_config(source: RunConfig, pipeline: RelabelPipeline, seed_offset: int) -> RunConfig:
    """The source's configuration with the pipeline's backend and mitigation."""
    data = source.model_dump(mode="json")
    data |= {
        "name": f"{source.name}-relabel-{pipeline.label}",
        "seed": source.seed + seed_offset,
        "backend": pipeline.backend.model_dump(mode="json"),
        "mitigation": pipeline.mitigation.model_dump(mode="json"),
    }
    data["optimizer"] = {**data["optimizer"], "measure_current": True}
    return RunConfig.model_validate(data)


def _cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.asarray(np.sum(a * b, axis=1) / norms, dtype=float)


def run_relabel(
    cfg: RelabelConfig, *, study_dir: Path | None = None, only: list[str] | None = None
) -> Path:
    """Every (pipeline, source) not yet done in ``study_dir`` (a new directory if ``None``)."""
    from datetime import UTC, datetime

    from kagome_vqe.results import load_run

    sources = _sources(cfg)
    if study_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        study_dir = Path(cfg.output_dir) / f"{stamp}-{cfg.name}"
        study_dir.mkdir(parents=True)
        with (study_dir / "relabel.toml").open("wb") as fh:
            tomli_w.dump(cfg.model_dump(mode="json", exclude_none=True), fh)
        (study_dir / "sources.json").write_text(
            json.dumps([str(s) for s in sources], indent=2) + "\n"
        )
    for pipeline in cfg.pipelines:
        for source in sources:
            seed = load_run(source).config.seed
            name = f"{pipeline.label}-s{seed}"
            out = study_dir / "runs" / name
            if (out / "status.json").exists() or (only and not any(o in name for o in only)):
                continue
            relabel_source(pipeline, source, out, t_init=cfg.t_init, seed_offset=cfg.seed_offset)
    return study_dir
