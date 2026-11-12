"""Fixed-parameter noise analysis: saved optimal parameters under PRB 106 layer noise.

This is the methodology of Kattemölle and van Wezel, PRB 106, 214429 (2022), Secs. II E and
III B: parameters are optimised *without* noise, then the energy and ground-state fidelity of
those fixed parameters are computed with local noise, and the infidelity is compared with the
estimate of their Eq. 9,

    F(theta*) ~ (1 - p_e)^(n d) F_0(theta*),      d = |c| p + 1,

which assumes every error makes the state orthogonal to the ground space. Nothing is
re-optimised here. For each saved run and each (channel, p_e) the analysis computes

* the noise-free energy and ground-space fidelity ``F_0 = Tr(P_GS |psi><psi|)``;
* the exact noisy energy ``Tr(H rho)`` and fidelity ``Tr(P_GS rho)`` from the full density
  matrix (Aer, up to 12 qubits);
* the paper's estimator: ``N`` noise trajectories, each contributing its exact energy and
  fidelity, with percentile-bootstrap 95 % confidence intervals. ``N`` follows the paper's
  rule: enough trajectories that on average ``expected_error_trajectories`` (1024) contain an
  error (Sec. II E), so the interval never collapses because no error was sampled;
* optionally a hardware-like *measured* energy (``shots`` per measurement basis, every shot a
  trajectory plus a projective measurement) with its standard error;
* optionally, as a separately labelled "compiled hardware noise" comparison, the energy through
  ``aer_noisy`` (a fake backend's noise model on the transpiled circuit).

Results go to ``results/<UTC timestamp>-<name>/`` with ``points.csv``/``.json``, ``report.md``
and figures, all regenerated from the saved points by :func:`summarise_noise_analysis`.
"""

from __future__ import annotations

import glob
import json
import logging
import math
import time
from collections.abc import Hashable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Self

import numpy as np
import pandas as pd
import tomli_w
from pydantic import BaseModel, ConfigDict, Field, model_validator

from kagome_vqe.ansatz import GraphHVAnsatz
from kagome_vqe.config import BackendConfig, MappingConfig, load_toml
from kagome_vqe.hamiltonian import exact_ground_space, heisenberg_hamiltonian, relative_error
from kagome_vqe.noise import (
    NoiseChannel,
    PauliTrajectorySimulator,
    exact_noisy_state,
    layer_noise_circuit,
    measurement_bases,
)

__all__ = [
    "HardwareComparison",
    "NoiseAnalysisConfig",
    "load_noise_analysis_config",
    "resolve_sources",
    "run_noise_analysis",
    "summarise_noise_analysis",
    "trajectories_for",
]

log = logging.getLogger(__name__)

_CHANNELS: tuple[NoiseChannel, ...] = ("depolarizing", "bit_flip")

EXACT_MAX_QUBITS = 12
"""Largest register for which the exact density matrix is computed (268 MB)."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


class HardwareComparison(_Frozen):
    """Compiled-hardware noise for the same parameters (``aer_noisy``), clearly separate.

    Attributes:
        label: Name in tables.
        backend: An ``aer_noisy`` backend configuration (fake backend, shots).
        mapping: Mapping used to compile the circuit for that backend.
        repeats: Independent energy estimates (each with a fresh simulator seed).
    """

    label: str
    backend: BackendConfig
    mapping: MappingConfig = MappingConfig()
    repeats: Annotated[int, Field(ge=1)] = 2

    @model_validator(mode="after")
    def _is_hardware_noise(self) -> Self:
        if self.backend.kind != "aer_noisy":
            msg = "hardware comparisons use backend.kind = 'aer_noisy' (no IBM jobs)"
            raise ValueError(msg)
        return self


class NoiseAnalysisConfig(_Frozen):
    """A fixed-parameter noise analysis (TOML in ``configs/noise/``).

    Attributes:
        name: Name of the output directory.
        description: Free text for the report.
        runs: Saved run directories or glob patterns (relative to the working directory, or
            to ``root``). Every run must use the ``graph_hva`` ansatz.
        root: Base directory for relative ``runs`` patterns; ``None`` means the working
            directory.
        channels: Channels of PRB 106 Eqs. 7-8.
        probabilities: Error probabilities ``p_e`` (the paper uses 1e-2 ... 1e-5).
        exact: Compute the exact density matrix where the register has at most 12 qubits.
        expected_error_trajectories: PRB 106's rule: sample so that on average this many
            trajectories contain at least one error.
        min_trajectories: Lower bound on the number of trajectories.
        max_trajectories: Upper bound (cost guard).
        shots: Shots per measurement basis for a hardware-like measured energy; 0 skips it.
        bootstrap: Bootstrap resamples for the confidence intervals.
        seed: Seeds every random stream of the analysis.
        hardware: Optional compiled-hardware comparisons (``aer_noisy``), reported separately.
        match_source_noise: Evaluate a run that was itself optimised under paper-layer noise
            only at its own (channel, ``p_e``); noise-free runs at every grid point. This is the
            transfer comparison: noise-free parameters carried over versus parameters
            re-optimised under the same noise.
    """

    name: str = "graph_hva_noise_fixed"
    description: str = ""
    runs: list[str]
    root: Path | None = None
    channels: list[NoiseChannel] = Field(default_factory=lambda: list(_CHANNELS))
    probabilities: list[Annotated[float, Field(ge=0, le=1)]] = Field(
        default_factory=lambda: [1e-2, 1e-3, 1e-4, 1e-5]
    )
    exact: bool = True
    expected_error_trajectories: Annotated[int, Field(ge=1)] = 1024
    min_trajectories: Annotated[int, Field(ge=1)] = 1024
    max_trajectories: Annotated[int, Field(ge=1)] = 10_000_000
    shots: Annotated[int, Field(ge=0)] = 0
    bootstrap: Annotated[int, Field(ge=100)] = 2000
    seed: int = 1
    hardware: list[HardwareComparison] = Field(default_factory=list)
    match_source_noise: bool = False


def load_noise_analysis_config(path: Path) -> NoiseAnalysisConfig:
    """Read an analysis TOML file."""
    return NoiseAnalysisConfig.model_validate(load_toml(Path(path)))


def trajectories_for(cfg: NoiseAnalysisConfig, p: float, locations: int) -> int:
    """PRB 106's trajectory count: ``expected / P(at least one error)``, within bounds."""
    p_error = 1.0 - (1.0 - p) ** locations
    wanted = cfg.min_trajectories if p_error <= 0 else cfg.expected_error_trajectories / p_error
    return int(min(cfg.max_trajectories, max(cfg.min_trajectories, math.ceil(wanted))))


def resolve_sources(cfg: NoiseAnalysisConfig) -> list[Path]:
    """Run directories matched by ``cfg.runs`` that hold a completed ``result.json``."""
    base = Path.cwd() if cfg.root is None else Path(cfg.root)
    found: list[Path] = []
    for pattern in cfg.runs:
        full = pattern if Path(pattern).is_absolute() else str(base / pattern)
        for match in sorted(glob.glob(full)):
            path = Path(match)
            if (path / "result.json").exists() and path not in found:
                found.append(path)
    if not found:
        msg = f"no completed runs match {cfg.runs} (relative to {base})"
        raise FileNotFoundError(msg)
    return found


def _now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _new_dir(cfg: NoiseAnalysisConfig, root: Path) -> Path:
    base = Path(root) / f"{_now()}-{cfg.name}"
    out, counter = base, 1
    while out.exists():
        out = base.with_name(f"{base.name}-{counter}")
        counter += 1
    out.mkdir(parents=True)
    return out


def _source_row(run: Any) -> dict[str, Any]:
    cfg = run.config
    return {
        "source": run.name,
        "source_combo": run.combo,
        "source_path": str(run.path),
        "source_backend": cfg.backend.label(),
        "source_optimizer": cfg.optimizer.label(),
        "lattice": cfg.lattice.kind
        + ("" if cfg.lattice.rows is None else f"_{cfg.lattice.rows}x{cfg.lattice.cols}"),
        "reps": cfg.ansatz.reps,
        "include_h0": cfg.ansatz.include_h0,
        "seed": cfg.seed,
    }


def _analyse_run(
    cfg: NoiseAnalysisConfig, run_dir: Path, rng: np.random.Generator
) -> list[dict[str, Any]]:
    from kagome_vqe.results import load_optimal_params, load_run

    run = load_run(run_dir)
    if run.config.ansatz.type != "graph_hva":
        msg = f"{run_dir}: fixed-parameter layer noise needs a graph_hva run"
        raise ValueError(msg)
    lattice = run.config.lattice.build()
    hamiltonian = heisenberg_hamiltonian(lattice)
    ansatz = run.config.ansatz.build(lattice)
    assert isinstance(ansatz, GraphHVAnsatz)
    theta = load_optimal_params(run_dir, num_parameters=ansatz.num_parameters)
    e0, ground = exact_ground_space(hamiltonian)
    h_matrix = hamiltonian.to_matrix(sparse=True)
    n, d = ansatz.num_qubits, ansatz.noise_layers
    clean = PauliTrajectorySimulator(layer_noise_circuit(ansatz, "depolarizing", 0.0))
    psi = clean.ideal_state(theta)
    ideal_energy = float(np.real(np.vdot(psi, h_matrix @ psi)))
    ideal_fidelity = float(np.sum(np.abs(ground.conj().T @ psi) ** 2))
    rows = []
    source_noise = run.config.backend.logical_noise
    for channel in cfg.channels:
        for p in cfg.probabilities:
            if (
                cfg.match_source_noise
                and source_noise is not None
                and (source_noise.channel, source_noise.probability) != (channel, p)
            ):
                continue
            tick = time.perf_counter()
            circuit = layer_noise_circuit(ansatz, channel, p)
            sim = PauliTrajectorySimulator(circuit)
            survival = (1.0 - p) ** (n * d)
            row: dict[str, Any] = {
                **_source_row(run),
                "num_qubits": n,
                "noise_layers": d,
                "locations": n * d,
                "channel": channel,
                "p": p,
                "exact_energy": e0,
                "ideal_energy": ideal_energy,
                "ideal_relative_error": relative_error(e0, ideal_energy),
                "ideal_fidelity": ideal_fidelity,
                "survival": survival,
                "eq9_fidelity": survival * ideal_fidelity,
                "eq9_infidelity": 1.0 - survival * ideal_fidelity,
            }
            if cfg.exact and n <= EXACT_MAX_QUBITS:
                rho = exact_noisy_state(circuit, theta)
                energy = float(np.real(np.trace(h_matrix @ rho)))
                fidelity = float(np.real(np.trace(ground.conj().T @ rho @ ground)))
                row |= {
                    "exact_noisy_energy": energy,
                    "exact_noisy_relative_error": relative_error(e0, energy),
                    "exact_energy_bias": energy - ideal_energy,
                    "exact_noisy_fidelity": fidelity,
                    "exact_noisy_infidelity": 1.0 - fidelity,
                    "eq9_ratio": (1.0 - fidelity) / (1.0 - survival * ideal_fidelity)
                    if survival * ideal_fidelity < 1
                    else math.nan,
                }
            n_traj = trajectories_for(cfg, p, n * d)
            sample = sim.sample_expectations(theta, [h_matrix, ground], n_traj, rng)
            mean = sample.mean()
            ci = sample.bootstrap_ci(rng, resamples=cfg.bootstrap)
            row |= {
                "trajectories": sample.trajectories,
                "error_free_trajectories": sample.error_free,
                "traj_energy": float(mean[0]),
                "traj_energy_ci_low": float(ci[0, 0]),
                "traj_energy_ci_high": float(ci[0, 1]),
                "traj_relative_error": relative_error(e0, float(mean[0])),
                "traj_fidelity": float(mean[1]),
                "traj_fidelity_ci_low": float(ci[1, 0]),
                "traj_fidelity_ci_high": float(ci[1, 1]),
                "traj_infidelity": 1.0 - float(mean[1]),
            }
            if cfg.shots:
                bases = measurement_bases(hamiltonian)
                histograms = sim.sample_measurements(theta, [b for b, _ in bases], cfg.shots, rng)
                measured, variance = 0.0, 0.0
                for counts, (_, diag) in zip(histograms, bases, strict=True):
                    first = float(counts @ diag) / cfg.shots
                    measured += first
                    variance += max(float(counts @ diag**2) / cfg.shots - first**2, 0) / cfg.shots
                row |= {
                    "shots_per_basis": cfg.shots,
                    "measured_energy": measured,
                    "measured_std": math.sqrt(variance),
                    "measured_relative_error": relative_error(e0, measured),
                }
            row["wall_time_s"] = time.perf_counter() - tick
            log.info(
                "%s %s p=%g: F0=%.4f F=%.4f (Eq. 9: %.4f)",
                run.name,
                channel,
                p,
                ideal_fidelity,
                row.get("exact_noisy_fidelity", row["traj_fidelity"]),
                row["eq9_fidelity"],
            )
            rows.append(row)
    return rows


def _hardware_rows(
    cfg: NoiseAnalysisConfig, run_dir: Path, settings: Any | None
) -> list[dict[str, Any]]:
    """Compiled-hardware comparison through ``aer_noisy`` (the existing fake-backend path)."""
    from kagome_vqe.results import load_optimal_params, load_run
    from kagome_vqe.vqe import evaluate_fixed

    run = load_run(run_dir)
    rows = []
    for hw in cfg.hardware:
        hw_cfg = run.config.model_copy(
            update={
                "name": f"{run.name}-{hw.label}",
                "backend": hw.backend,
                "mapping": hw.mapping,
                "budget": run.config.budget.model_copy(update={"max_shots": None}),
            }
        )
        theta = load_optimal_params(run_dir)
        tick = time.perf_counter()
        fixed = evaluate_fixed(hw_cfg, theta, repeats=hw.repeats, settings=settings)
        summary = fixed.summary()
        isa = summary["estimator"]
        rows.append(
            {
                **_source_row(run),
                "hardware": hw.label,
                "backend": hw.backend.label(),
                "mapping": hw.mapping.strategy,
                "ideal_energy": fixed.ideal_energy,
                "exact_energy": fixed.exact_energy,
                "mean_energy": fixed.mean,
                "spread": fixed.spread,
                "bias": fixed.bias,
                "relative_error": relative_error(fixed.exact_energy, fixed.mean),
                "repeats": hw.repeats,
                "shots_per_evaluation": fixed.shots_per_evaluation,
                "estimator": json.dumps(isa, default=str),
                "wall_time_s": time.perf_counter() - tick,
            }
        )
    return rows


def run_noise_analysis(
    cfg: NoiseAnalysisConfig,
    *,
    output_root: Path = Path("results"),
    settings: Any | None = None,
) -> Path:
    """Run the analysis into a new directory and write its summary.

    Returns:
        The output directory.
    """
    from kagome_vqe.results import environment_info

    sources = resolve_sources(cfg)
    out = _new_dir(cfg, output_root)
    with (out / "analysis.toml").open("wb") as fh:
        tomli_w.dump(cfg.model_dump(mode="json", exclude_none=True), fh)
    (out / "environment.json").write_text(json.dumps(environment_info(), indent=2) + "\n")
    (out / "sources.json").write_text(json.dumps([str(s) for s in sources], indent=2) + "\n")
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, Any]] = []
    hardware: list[dict[str, Any]] = []
    for source in sources:
        rows.extend(_analyse_run(cfg, source, rng))
        pd.DataFrame(rows).to_csv(out / "points.csv", index=False)  # keep partial progress
        if cfg.hardware:
            hardware.extend(_hardware_rows(cfg, source, settings))
            pd.DataFrame(hardware).to_csv(out / "hardware.csv", index=False)
    (out / "points.json").write_text(json.dumps(rows, indent=2, default=float) + "\n")
    summarise_noise_analysis(out)
    return out


# -- summary ------------------------------------------------------------------------------------


def _fmt(value: Any, spec: str = ".3e") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return format(value, spec)


def _source_label(row: Mapping[Hashable, Any]) -> str:
    backend = str(row["source_backend"])
    trained = "noise-free" if backend == "statevector" else backend
    return f"{row['source_optimizer']} ({trained}) s{row['seed']}"


def _aggregate(points: pd.DataFrame) -> list[str]:
    """Mean over seeds per (source backend, optimiser, p, channel, p_e)."""
    if "exact_noisy_relative_error" not in points:
        return []
    keys = ["source_combo", "source_backend", "reps", "channel", "p"]
    lines = [
        "",
        "## Mean over seeds",
        "",
        "Per source (where the parameters were optimised) and noise model. `transfer`: noise-free "
        "parameters evaluated under noise; a logical-noise source was optimised under that noise.",
        "",
        "| source (run name without seed) | optimised on | p | channel | p_e | seeds "
        "| mean E_ideal rel. err "
        "| mean E noisy rel. err (exact) | mean F0 | mean F noisy (exact) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, group in points.groupby(keys, sort=True):
        combo, backend, reps, channel, p = key
        lines.append(
            f"| {combo} | {backend} | {reps} | {channel} | {p:g} | {len(group)} "
            f"| {_fmt(group['ideal_relative_error'].mean())} "
            f"| {_fmt(group['exact_noisy_relative_error'].mean())} "
            f"| {_fmt(group['ideal_fidelity'].mean(), '.4f')} "
            f"| {_fmt(group['exact_noisy_fidelity'].mean(), '.4f')} |"
        )
    return lines


def _report(points: pd.DataFrame, hardware: pd.DataFrame | None, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "Fixed-parameter noise analysis (PRB 106 Secs. II E, III B): parameters optimised "
        "without noise, evaluated under local noise on every qubit after the singlet "
        "preparation and after every exchange layer. Nothing re-optimised.",
        "",
        "* `F0`, `E_ideal`: noise-free fidelity with the ground space and energy.",
        "* `F exact`, `E exact`: from the exact density matrix (`Tr(P_GS rho)`, `Tr(H rho)`).",
        "* `F traj`: PRB 106 estimator (mean over noise trajectories, 95 % bootstrap CI).",
        "* `Eq. 9`: `(1 - p)^(n d) F0`; `ratio` = simulated infidelity / Eq. 9 infidelity.",
        "",
        "| source | p | channel | p_e | n d | E_ideal rel. err | E noisy rel. err "
        "| bias | F0 | F exact | F traj [95 % CI] | Eq. 9 F | ratio |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in points.to_dict("records"):
        energy_err = r.get("exact_noisy_relative_error", r.get("traj_relative_error"))
        bias = r.get("exact_energy_bias", r["traj_energy"] - r["ideal_energy"])
        lines.append(
            f"| {_source_label(r)} | {r['reps']} | {r['channel']} "
            f"| {r['p']:g} | {r['locations']} | {_fmt(r['ideal_relative_error'])} "
            f"| {_fmt(energy_err)} | {_fmt(bias, '+.4f')} | {_fmt(r['ideal_fidelity'], '.4f')} "
            f"| {_fmt(r.get('exact_noisy_fidelity'), '.4f')} "
            f"| {_fmt(r['traj_fidelity'], '.4f')} [{_fmt(r['traj_fidelity_ci_low'], '.4f')}, "
            f"{_fmt(r['traj_fidelity_ci_high'], '.4f')}] | {_fmt(r['eq9_fidelity'], '.4f')} "
            f"| {_fmt(r.get('eq9_ratio'), '.3f')} |"
        )
    lines += _aggregate(points)
    if hardware is not None and not hardware.empty:
        lines += [
            "",
            "## Compiled hardware noise (separate mode: fake-backend noise model on the "
            "transpiled circuit)",
            "",
            "| source | p | hardware | mapping | E_ideal | mean E | spread | rel. err |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in hardware.to_dict("records"):
            lines.append(
                f"| {r['source_optimizer']} s{r['seed']} | {r['reps']} | {r['backend']} "
                f"| {r['mapping']} | {r['ideal_energy']:.4f} | {r['mean_energy']:.4f} "
                f"| {r['spread']:.4f} | {_fmt(r['relative_error'])} |"
            )
    return "\n".join(lines) + "\n"


def summarise_noise_analysis(out_dir: Path) -> Path:
    """(Re)write ``report.md`` and the figures from ``points.csv`` (and ``hardware.csv``)."""
    from kagome_vqe.plotting import plot_noise_analysis

    out_dir = Path(out_dir)
    points = pd.read_csv(out_dir / "points.csv")
    if "source_combo" not in points:  # analyses written before the column existed
        points["source_combo"] = points["source"].str.replace(r"-s\d+$", "", regex=True)
    hw_path = out_dir / "hardware.csv"
    hardware = pd.read_csv(hw_path) if hw_path.exists() else None
    title = out_dir.name
    analysis = out_dir / "analysis.toml"
    if analysis.exists():
        title = str(load_toml(analysis).get("name", title))
    (out_dir / "report.md").write_text(_report(points, hardware, title))
    plot_noise_analysis(points, out_dir / "figures")
    return out_dir / "report.md"
