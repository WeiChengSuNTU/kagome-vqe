"""M2-style mitigation table for ``graph_hva`` under PRB 106 layer noise (fixed parameters).

The legacy M2 study (``docs/experiments/legacy_m0_m1_m2.md``) compared readout correction,
symmetry filtering, ZNE and twin rescaling, alone and combined, by estimating the energy at fixed
parameters many times through each stack and reporting the bias of the mean against the
noise-free energy, the spread between repeats and the shot cost. This module repeats exactly that
protocol for the graph HVA on the logical-layer noise model (optionally with a readout error,
``logical_noise.readout_error``), so the techniques can be compared on the noise model the
graph_hva studies use.

Every repeat builds a fresh estimator (its own seed, readout calibration and twin), so one-off
measurements contribute their noise to the spread, as in the corrected M2 protocol.

Outputs (``results/<UTC timestamp>-<name>/``): ``points.csv`` / ``.json`` (one row per point,
noise model and stack) and ``report.md`` (the M2 table: bias in % of the noise-free energy with
the standard error of the mean, spread, cost).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Annotated, Any, Self

import numpy as np
import pandas as pd
import tomli_w
from pydantic import BaseModel, ConfigDict, Field, model_validator

from kagome_vqe.config import (
    AnsatzConfig,
    BackendConfig,
    LatticeConfig,
    MitigationConfig,
    RunConfig,
    load_toml,
)
from kagome_vqe.noise_analysis import _new_dir

__all__ = [
    "MitigationStudyConfig",
    "load_mitigation_study_config",
    "run_mitigation_study",
    "summarise_mitigation_study",
]

log = logging.getLogger(__name__)

_LABEL = r"^[A-Za-z0-9_.+]+$"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


class StudyPoint(_Frozen):
    """A parameter point: a saved run's final parameters or one value for every parameter.

    Attributes:
        label: Name in tables.
        run: Run directory whose optimal parameters are used (same lattice and ansatz).
        value: Every parameter set to this value (e.g. 0.3, a generic state).
    """

    label: Annotated[str, Field(pattern=_LABEL)]
    run: str | None = None
    value: float | None = None

    @model_validator(mode="after")
    def _one_source(self) -> Self:
        if (self.run is None) == (self.value is None):
            msg = "a point needs exactly one of run / value"
            raise ValueError(msg)
        return self


class StudyNoise(_Frozen):
    """A noise model: a ``logical_noise`` backend (channel, ``p_e``, readout error, shots)."""

    label: Annotated[str, Field(pattern=_LABEL)]
    backend: BackendConfig


class StudyStack(_Frozen):
    """A mitigation stack (counts-level, then ZNE, then twin rescaling)."""

    label: Annotated[str, Field(pattern=_LABEL)]
    mitigation: MitigationConfig = MitigationConfig()


class MitigationStudyConfig(_Frozen):
    """The M2-style fixed-point comparison (TOML in ``configs/mitigation/``).

    Attributes:
        name: Output directory suffix.
        description: Free text for the report.
        seed: Seeds the repeats (and the automatic twin search).
        lattice: The lattice.
        ansatz: The ansatz (``graph_hva``).
        points: Parameter points.
        noise: Noise models.
        stacks: Mitigation stacks.
        repeats: Fresh-setup estimates per cell.
        root: Base directory for relative ``points.run`` paths.
    """

    name: str = "graph_hva_m2"
    description: str = ""
    seed: int = 42
    lattice: LatticeConfig
    ansatz: AnsatzConfig
    points: Annotated[list[StudyPoint], Field(min_length=1)]
    noise: Annotated[list[StudyNoise], Field(min_length=1)]
    stacks: Annotated[list[StudyStack], Field(min_length=1)]
    repeats: Annotated[int, Field(ge=2)] = 16
    root: Path | None = None
    output_dir: Path = Path("results")

    def cells(self) -> list[tuple[StudyPoint, StudyNoise, StudyStack, RunConfig]]:
        """Every (point, noise, stack) with its run configuration."""
        out = []
        for point in self.points:
            for noise in self.noise:
                for stack in self.stacks:
                    cfg = RunConfig(
                        name=f"{self.name}-{point.label}-{noise.label}-{stack.label}",
                        seed=self.seed,
                        lattice=self.lattice,
                        ansatz=self.ansatz,
                        backend=noise.backend,
                        mitigation=stack.mitigation,
                        track_ideal_energy=False,
                    )
                    out.append((point, noise, stack, cfg))
        return out


def load_mitigation_study_config(path: Path) -> MitigationStudyConfig:
    """Read a study TOML."""
    return MitigationStudyConfig.model_validate(load_toml(Path(path)))


def _theta(cfg: MitigationStudyConfig, point: StudyPoint, num_parameters: int) -> Any:
    if point.value is not None:
        return float(point.value)
    from kagome_vqe.results import load_optimal_params

    assert point.run is not None
    base = Path.cwd() if cfg.root is None else Path(cfg.root)
    path = Path(point.run) if Path(point.run).is_absolute() else base / point.run
    return load_optimal_params(path, num_parameters=num_parameters)


def _evaluate_cell(
    cfg: MitigationStudyConfig,
    point: StudyPoint,
    noise: StudyNoise,
    stack: StudyStack,
    run_cfg: RunConfig,
) -> dict[str, Any]:
    from kagome_vqe.vqe import evaluate_fixed, prepare

    prep = prepare(run_cfg)
    theta = _theta(cfg, point, prep.ansatz.num_parameters)
    result = evaluate_fixed(run_cfg, theta, repeats=cfg.repeats, prepared=prep)
    sem = result.spread / math.sqrt(result.energies.size)
    row = {
        "point": point.label,
        "noise": noise.label,
        "stack": stack.label,
        "mitigation": run_cfg.mitigation.label(),
        "backend": run_cfg.backend.label(),
        "ideal_energy": result.ideal_energy,
        "exact_energy": result.exact_energy,
        "mean": result.mean,
        "bias": result.bias,
        "bias_percent": 100 * result.bias / abs(result.ideal_energy),
        "bias_sem_percent": 100 * sem / abs(result.ideal_energy),
        "spread": result.spread,
        "mean_reported_std": float(np.mean(result.stds)),
        "rmse": math.hypot(result.bias, result.spread),
        "below_ground": result.below_ground,
        "repeats": int(result.energies.size),
        "shots_per_energy": result.shots_per_evaluation,
        "setup_shots": result.setup_shots,
        "wall_time_s": result.wall_time_s,
        "energies": json.dumps([float(e) for e in result.energies]),
    }
    log.info(
        "%s: bias %+.2f%% +- %.2f%%, spread %.4f",
        run_cfg.name,
        row["bias_percent"],
        row["bias_sem_percent"],
        result.spread,
    )
    return row


def run_mitigation_study(
    cfg: MitigationStudyConfig,
    *,
    output_root: Path | None = None,
    only: list[str] | None = None,
    workers: int = 1,
) -> Path:
    """Evaluate every cell (or those whose name contains one of ``only``).

    ``workers > 1`` evaluates cells in parallel processes; rows are written in cell order, so
    the output does not depend on it.
    """
    from concurrent.futures import ProcessPoolExecutor

    from kagome_vqe.results import environment_info

    out = _new_dir(cfg, Path(output_root or cfg.output_dir))
    with (out / "study.toml").open("wb") as fh:
        tomli_w.dump(cfg.model_dump(mode="json", exclude_none=True), fh)
    (out / "environment.json").write_text(json.dumps(environment_info(), indent=2) + "\n")
    cells = [c for c in cfg.cells() if not only or any(o in c[3].name for o in only)]
    rows: list[dict[str, Any]] = []
    if workers <= 1:
        for cell in cells:
            rows.append(_evaluate_cell(cfg, *cell))
            pd.DataFrame(rows).to_csv(out / "points.csv", index=False)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_evaluate_cell, cfg, *cell) for cell in cells]
            for future in futures:
                rows.append(future.result())
                pd.DataFrame(rows).to_csv(out / "points.csv", index=False)
    (out / "points.json").write_text(json.dumps(rows, indent=2, default=float) + "\n")
    summarise_mitigation_study(out)
    return out


def summarise_mitigation_study(out_dir: Path) -> Path:
    """(Re)write ``report.md`` from ``points.csv``."""
    out_dir = Path(out_dir)
    frame = pd.read_csv(out_dir / "points.csv")
    title = out_dir.name
    if (out_dir / "study.toml").exists():
        title = str(load_toml(out_dir / "study.toml").get("name", title))
    columns = list(dict.fromkeys(zip(frame["noise"], frame["point"], strict=True)))
    lines = [
        f"# {title}",
        "",
        "Bias of the mean against the noise-free energy of the same parameters, in % of it,",
        "+- the standard error of the mean over fresh-setup repeats (one-off calibration and",
        "twin redone each time). Spread = standard deviation between repeats (the real error of",
        "one estimate). Cost: circuit shots per energy, one-off shots per setup. ⚠: mean",
        "significantly below the exact ground energy (an artefact).",
        "",
        "| stack | "
        + " | ".join(f"{n}, {p}" for n, p in columns)
        + " | spread (first column) | shots per energy | one-off shots |",
        "|---|" + "---|" * len(columns) + "---|---|---|",
    ]
    for stack, sub in frame.groupby("stack", sort=False):
        cells = []
        for noise, point in columns:
            row = sub[(sub["noise"] == noise) & (sub["point"] == point)]
            if row.empty:
                cells.append("-")
                continue
            r = row.iloc[0]
            flag = " ⚠" if bool(r["below_ground"]) else ""
            cells.append(f"{r['bias_percent']:+.2f} ± {r['bias_sem_percent']:.2f}{flag}")
        first = sub.iloc[0]
        lines.append(
            f"| {stack} | "
            + " | ".join(cells)
            + f" | {first['spread']:.3f} | {int(first['shots_per_energy']):,} | "
            f"{int(first['setup_shots']):,} |"
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    return out_dir / "report.md"
