"""Fixed-parameter zero-noise extrapolation under PRB 106 layer noise: bias, variance, cost.

Before ZNE is used inside an optimiser it has to be characterised where the answer is known.
This analysis takes saved parameters (noise-free optima, or any run), and for every
``(channel, p_e)`` computes from exact density matrices

* the noisy energy ``E(lambda)`` at every noise-scale factor of every configured method:
  ``probability`` scaling (``p_e -> lambda p_e`` at every location) and ``layer_folding``
  (``L -> L (L^dag L)^k``, ``lambda = 2k + 1``, see :mod:`kagome_vqe.noise`);
* the shot-noise variance of a ``shots``-per-basis measurement of each ``E(lambda)``, from the
  exact per-group variances ``Tr(rho H_g^2) - Tr(rho H_g)^2``;
* per method the zero-noise estimate an *infinite-shot* experiment would get (its remaining
  ``bias`` against the noise-free energy of the same parameters, which isolates the
  extrapolation error), the standard error a ``shots`` experiment would have, and the
  root-mean-square error ``sqrt(bias^2 + se^2)``, next to the unmitigated values.

Optionally (``sampled_repeats``) the complete measured pipeline is repeated with the
trajectory sampler, to check the predicted spread against independent sampled ZNE estimates.

Nothing is optimised here. Outputs go to ``results/<UTC timestamp>-<name>/``: ``levels.csv``
(one row per noise level), ``points.csv`` (one row per method), ``report.md``, figures.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import numpy as np
import pandas as pd
import tomli_w
from pydantic import BaseModel, ConfigDict, Field, model_validator

from kagome_vqe.ansatz import GraphHVAnsatz
from kagome_vqe.config import load_toml
from kagome_vqe.hamiltonian import heisenberg_hamiltonian
from kagome_vqe.mitigation.wrappers import Extrapolation, extrapolate
from kagome_vqe.noise import (
    NoiseChannel,
    PauliTrajectorySimulator,
    exact_noisy_state,
    fold_noise_layers,
    layer_noise_circuit,
    measurement_bases,
    scale_noise_probability,
)
from kagome_vqe.noise_analysis import _new_dir, _source_row, resolve_sources

__all__ = [
    "ZneAnalysisConfig",
    "ZneMethod",
    "load_zne_analysis_config",
    "resolve_sources",
    "run_zne_analysis",
    "summarise_zne_analysis",
]

log = logging.getLogger(__name__)

EXACT_MAX_QUBITS = 12

_DEFAULT_CHANNELS: tuple[NoiseChannel, ...] = ("depolarizing",)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


class ZneMethod(_Frozen):
    """One way of doing ZNE.

    Attributes:
        label: Name in tables (letters, digits, ``_``, ``.``).
        noise_scaling: ``"probability"`` or ``"layer_folding"``.
        scales: Noise-scale factors, starting with 1 (odd integers for folding).
        extrapolation: Fit model of :func:`kagome_vqe.mitigation.extrapolate`.
    """

    label: Annotated[str, Field(pattern=r"^[A-Za-z0-9_.]+$")]
    noise_scaling: Literal["probability", "layer_folding"]
    scales: list[Annotated[float, Field(ge=1)]]
    extrapolation: Extrapolation

    @model_validator(mode="after")
    def _valid(self) -> Self:
        if len(self.scales) < 2 or len(set(self.scales)) != len(self.scales):
            msg = "a ZNE method needs at least two distinct scales"
            raise ValueError(msg)
        if self.scales[0] != 1:
            msg = "ZNE scales must start with 1"
            raise ValueError(msg)
        if self.noise_scaling == "layer_folding" and any(
            not s.is_integer() or int(s) % 2 == 0 for s in self.scales
        ):
            msg = "layer_folding scales must be odd integers"
            raise ValueError(msg)
        return self


class ZneAnalysisConfig(_Frozen):
    """A fixed-parameter ZNE characterisation (TOML in ``configs/mitigation/``).

    Attributes:
        name: Output directory suffix.
        description: Free text for the report.
        runs: Saved ``graph_hva`` run directories or glob patterns.
        root: Base directory for relative ``runs``.
        channels: PRB 106 channels.
        probabilities: Base error probabilities ``p_e``.
        methods: ZNE methods to compare.
        shots: Shots per measurement basis and noise scale assumed for standard errors.
        sampled_repeats: Independent sampled ZNE estimates per method (trajectory sampler,
            ``shots`` per basis); 0 skips the sampled check.
        seed: Seeds the sampled check.
    """

    name: str = "graph_hva_zne_fixed"
    description: str = ""
    runs: list[str]
    root: Path | None = None
    channels: list[NoiseChannel] = Field(default_factory=lambda: list(_DEFAULT_CHANNELS))
    probabilities: list[Annotated[float, Field(gt=0, lt=1)]] = Field(default_factory=lambda: [1e-3])
    methods: Annotated[list[ZneMethod], Field(min_length=1)]
    shots: Annotated[int, Field(ge=1)] = 1024
    sampled_repeats: Annotated[int, Field(ge=0)] = 0
    seed: int = 1

    @model_validator(mode="after")
    def _unique_labels(self) -> Self:
        labels = [m.label for m in self.methods]
        if len(set(labels)) != len(labels):
            msg = f"duplicate method labels {labels}"
            raise ValueError(msg)
        return self

    def levels(self) -> list[tuple[str, float]]:
        """Distinct ``(noise_scaling, lambda)`` pairs; ``lambda = 1`` is shared."""
        seen: list[tuple[str, float]] = [("none", 1.0)]
        for method in self.methods:
            for scale in method.scales:
                key = ("none", 1.0) if scale == 1 else (method.noise_scaling, float(scale))
                if key not in seen:
                    seen.append(key)
        return seen


def load_zne_analysis_config(path: Path) -> ZneAnalysisConfig:
    """Read an analysis TOML file."""
    return ZneAnalysisConfig.model_validate(load_toml(Path(path)))


def _scaled_circuit(base: Any, scaling: str, factor: float) -> Any:
    if scaling == "none" or factor == 1:
        return base
    if scaling == "probability":
        return scale_noise_probability(base, factor)
    return fold_noise_layers(base, (round(factor) - 1) // 2)


def _expectation(operator: Any, rho: np.ndarray) -> float:
    """``Tr(A rho)`` for a sparse ``A``."""
    return float(np.real(operator.multiply(rho.T).sum()))


def _level_rows(
    cfg: ZneAnalysisConfig, run: Any, ansatz: GraphHVAnsatz, theta: np.ndarray
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Exact energies and shot variances at every noise level; plus noise-free reference."""
    hamiltonian = heisenberg_hamiltonian(run.config.lattice.build())
    h_matrix = hamiltonian.to_matrix(sparse=True)
    groups = [g.to_matrix(sparse=True) for g in hamiltonian.group_commuting(qubit_wise=True)]
    squares = [g @ g for g in groups]
    clean = PauliTrajectorySimulator(layer_noise_circuit(ansatz, "depolarizing", 0.0))
    psi = clean.ideal_state(theta)
    ideal = float(np.real(np.vdot(psi, h_matrix @ psi)))
    reference = {"ideal_energy": ideal}
    rows = []
    for channel in cfg.channels:
        for p in cfg.probabilities:
            base = layer_noise_circuit(ansatz, channel, p)
            for scaling, factor in cfg.levels():
                tick = time.perf_counter()
                circuit = _scaled_circuit(base, scaling, factor)
                rho = exact_noisy_state(circuit, theta)
                energy = _expectation(h_matrix, rho)
                variance = sum(
                    max(_expectation(sq, rho) - _expectation(g, rho) ** 2, 0.0)
                    for g, sq in zip(groups, squares, strict=True)
                )
                rows.append(
                    {
                        **_source_row(run),
                        "channel": channel,
                        "p": p,
                        "noise_scaling": scaling,
                        "scale": factor,
                        "energy": energy,
                        "ideal_energy": ideal,
                        "bias": energy - ideal,
                        "shot_variance_one_shot": variance,
                        "se": math.sqrt(variance / cfg.shots),
                        "gates": circuit.size(),
                        "wall_time_s": time.perf_counter() - tick,
                    }
                )
                log.info(
                    "%s %s p=%g %s x%g: E=%.5f (ideal %.5f)",
                    run.name,
                    channel,
                    p,
                    scaling,
                    factor,
                    energy,
                    ideal,
                )
                del rho
    return rows, reference


def _method_rows(
    cfg: ZneAnalysisConfig, levels: pd.DataFrame, exact_energy: float
) -> list[dict[str, Any]]:
    rows = []
    for (channel, p), frame in levels.groupby(["channel", "p"], sort=False):
        base = frame[frame["noise_scaling"] == "none"].iloc[0]
        ideal = float(base["ideal_energy"])
        unmitigated_bias = float(base["bias"])
        unmitigated_se = float(base["se"])
        for method in cfg.methods:
            values, stds = [], []
            for scale in method.scales:
                kind = "none" if scale == 1 else method.noise_scaling
                level = frame[(frame["noise_scaling"] == kind) & (frame["scale"] == scale)]
                values.append(float(level["energy"].iloc[0]))
                stds.append(float(level["se"].iloc[0]))
            estimate, se = extrapolate(
                method.scales, np.array(values), np.array(stds), method.extrapolation
            )
            bias = estimate - ideal
            depth = (
                sum(method.scales)
                if method.noise_scaling == "layer_folding"
                else len(method.scales)
            )
            rows.append(
                {
                    **{
                        k: base[k]
                        for k in base.index
                        if k.startswith("source") or k in ("lattice", "reps", "include_h0", "seed")
                    },
                    "channel": channel,
                    "p": p,
                    "method": method.label,
                    "noise_scaling": method.noise_scaling,
                    "scales": " ".join(f"{s:g}" for s in method.scales),
                    "extrapolation": method.extrapolation,
                    "exact_energy": exact_energy,
                    "ideal_energy": ideal,
                    "unmitigated_energy": float(base["energy"]),
                    "unmitigated_bias": unmitigated_bias,
                    "unmitigated_se": unmitigated_se,
                    "unmitigated_rmse": math.hypot(unmitigated_bias, unmitigated_se),
                    "zne_energy": estimate,
                    "zne_bias": bias,
                    "zne_se": se,
                    "zne_rmse": math.hypot(bias, se),
                    "bias_ratio": abs(bias) / abs(unmitigated_bias)
                    if unmitigated_bias
                    else math.nan,
                    "se_amplification": se / unmitigated_se if unmitigated_se else math.nan,
                    "circuits_per_energy": len(method.scales),
                    "layer_cost_per_energy": depth,
                    "shots": cfg.shots,
                }
            )
    return rows


def _sampled_rows(
    cfg: ZneAnalysisConfig,
    run: Any,
    ansatz: GraphHVAnsatz,
    theta: np.ndarray,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Repeated sampled ZNE estimates (trajectory sampler, fast kernels)."""
    hamiltonian = heisenberg_hamiltonian(run.config.lattice.build())
    bases = measurement_bases(hamiltonian)
    letters = [b for b, _ in bases]
    rows = []
    for channel in cfg.channels:
        for p in cfg.probabilities:
            base = layer_noise_circuit(ansatz, channel, p)
            sims: dict[tuple[str, float], PauliTrajectorySimulator] = {}
            for method in cfg.methods:
                estimates = []
                for _ in range(cfg.sampled_repeats):
                    values, stds = [], []
                    for scale in method.scales:
                        kind = "none" if scale == 1 else method.noise_scaling
                        if (kind, scale) not in sims:
                            sims[kind, scale] = PauliTrajectorySimulator(
                                _scaled_circuit(base, kind, scale), fast=True
                            )
                        histograms = sims[kind, scale].sample_measurements(
                            theta, letters, cfg.shots, rng
                        )
                        mean, var = 0.0, 0.0
                        for counts, (_, diag) in zip(histograms, bases, strict=True):
                            first = float(counts @ diag) / cfg.shots
                            mean += first
                            var += max(float(counts @ diag**2) / cfg.shots - first**2, 0)
                        values.append(mean)
                        stds.append(math.sqrt(var / cfg.shots))
                    estimates.append(
                        extrapolate(
                            method.scales, np.array(values), np.array(stds), method.extrapolation
                        )[0]
                    )
                arr = np.asarray(estimates)
                rows.append(
                    {
                        "source": run.name,
                        "channel": channel,
                        "p": p,
                        "method": method.label,
                        "repeats": arr.size,
                        "sampled_mean": float(arr.mean()),
                        "sampled_std": float(arr.std(ddof=1)) if arr.size > 1 else math.nan,
                    }
                )
    return rows


def _analyse_source(
    cfg: ZneAnalysisConfig, source: Path, index: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Levels, method rows and sampled rows of one saved run (independent random stream)."""
    from kagome_vqe.hamiltonian import exact_spectrum
    from kagome_vqe.results import load_optimal_params, load_run

    run = load_run(source)
    if run.config.ansatz.type != "graph_hva":
        msg = f"{source}: layer-noise ZNE needs a graph_hva run"
        raise ValueError(msg)
    lattice = run.config.lattice.build()
    ansatz = run.config.ansatz.build(lattice)
    assert isinstance(ansatz, GraphHVAnsatz)
    if ansatz.num_qubits > EXACT_MAX_QUBITS:
        msg = f"{source}: exact density matrices are limited to {EXACT_MAX_QUBITS} qubits"
        raise ValueError(msg)
    theta = load_optimal_params(source, num_parameters=ansatz.num_parameters)
    exact = float(exact_spectrum(heisenberg_hamiltonian(lattice), k=1)[0])
    levels, _ = _level_rows(cfg, run, ansatz, theta)
    points = _method_rows(cfg, pd.DataFrame(levels), exact)
    sampled: list[dict[str, Any]] = []
    if cfg.sampled_repeats:
        rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, index]))
        sampled = _sampled_rows(cfg, run, ansatz, theta, rng)
    return levels, points, sampled


def run_zne_analysis(
    cfg: ZneAnalysisConfig, *, output_root: Path = Path("results"), workers: int = 1
) -> Path:
    """Run the analysis into a new directory and write its summary.

    Args:
        cfg: The analysis.
        output_root: Parent of the new directory.
        workers: Source runs analysed in parallel processes (each single-threaded work on
            one density matrix at a time); results do not depend on it.
    """
    from concurrent.futures import ProcessPoolExecutor

    from kagome_vqe.results import environment_info

    sources = resolve_sources(cfg)
    out = _new_dir(cfg, output_root)
    with (out / "analysis.toml").open("wb") as fh:
        tomli_w.dump(cfg.model_dump(mode="json", exclude_none=True), fh)
    (out / "environment.json").write_text(json.dumps(environment_info(), indent=2) + "\n")
    (out / "sources.json").write_text(json.dumps([str(s) for s in sources], indent=2) + "\n")
    levels: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    sampled: list[dict[str, Any]] = []

    def collect(result: tuple[list[dict[str, Any]], ...]) -> None:
        levels.extend(result[0])
        points.extend(result[1])
        sampled.extend(result[2])
        pd.DataFrame(levels).to_csv(out / "levels.csv", index=False)  # keep partial progress
        pd.DataFrame(points).to_csv(out / "points.csv", index=False)
        if sampled:
            pd.DataFrame(sampled).to_csv(out / "sampled.csv", index=False)

    if workers <= 1:
        for index, source in enumerate(sources):
            collect(_analyse_source(cfg, source, index))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_analyse_source, cfg, source, index)
                for index, source in enumerate(sources)
            ]
            for future in futures:  # submission order, so files do not depend on timing
                collect(future.result())
    summarise_zne_analysis(out)
    return out


# -- summary ------------------------------------------------------------------------------------


def _fmt(value: Any, spec: str = ".3g") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return format(value, spec)


def _report(points: pd.DataFrame, sampled: pd.DataFrame | None, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "Fixed parameters, exact density matrices. `bias` is the energy minus the noise-free",
        "energy of the same parameters (what mitigation should remove); `se` is the standard",
        f"error of a measurement with {int(points['shots'].iloc[0])} shots per basis and noise",
        "scale; `rmse = sqrt(bias^2 + se^2)`. Means over sources (seeds) per",
        "(p, channel, p_e, method).",
        "",
        "| p | channel | p_e | method | unmitigated bias | ZNE bias | bias ratio | "
        "unmitigated se | ZNE se | unmitigated rmse | ZNE rmse | circuits | layer cost |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    keys = ["reps", "channel", "p", "method"]
    agg = points.groupby(keys, sort=False).agg(
        unmitigated_bias=("unmitigated_bias", "mean"),
        zne_bias=("zne_bias", "mean"),
        bias_ratio=("bias_ratio", "mean"),
        unmitigated_se=("unmitigated_se", "mean"),
        zne_se=("zne_se", "mean"),
        unmitigated_rmse=("unmitigated_rmse", "mean"),
        zne_rmse=("zne_rmse", "mean"),
        circuits=("circuits_per_energy", "first"),
        layers=("layer_cost_per_energy", "first"),
    )
    for key, row in agg.iterrows():
        reps, channel, p, method = (str(k) for k in key)  # type: ignore[attr-defined]
        lines.append(
            f"| {reps} | {channel} | {float(p):g} | {method} | "
            f"{_fmt(row.unmitigated_bias, '+.4f')} | "
            f"{_fmt(row.zne_bias, '+.4f')} | {_fmt(row.bias_ratio, '.3f')} | "
            f"{_fmt(row.unmitigated_se, '.4f')} | {_fmt(row.zne_se, '.4f')} | "
            f"{_fmt(row.unmitigated_rmse, '.4f')} | {_fmt(row.zne_rmse, '.4f')} | "
            f"{int(row.circuits)} | {_fmt(row.layers, 'g')} |"
        )
    if sampled is not None and not sampled.empty:
        merged = sampled.merge(
            points[["source", "channel", "p", "method", "zne_energy", "zne_se"]],
            on=["source", "channel", "p", "method"],
            how="left",
        )
        lines += [
            "",
            "## Sampled check",
            "",
            "Independent sampled ZNE estimates (trajectory sampler) against the exact prediction.",
            "",
            "| source | channel | p_e | method | repeats | predicted mean | sampled mean | "
            "predicted se | sampled std |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for _, row in merged.iterrows():
            lines.append(
                f"| {row.source} | {row.channel} | {row.p:g} | {row.method} | {row.repeats} | "
                f"{row.zne_energy:.4f} | {row.sampled_mean:.4f} | {row.zne_se:.4f} | "
                f"{_fmt(row.sampled_std, '.4f')} |"
            )
    return "\n".join(lines) + "\n"


def _plot(levels: pd.DataFrame, points: pd.DataFrame, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    agg = (
        points.groupby(["reps", "channel", "p", "method"], sort=False)[
            ["unmitigated_bias", "zne_bias", "unmitigated_rmse", "zne_rmse"]
        ]
        .mean()
        .reset_index()
    )
    methods = list(dict.fromkeys(agg["method"]))
    colours = dict(zip(methods, plt.cm.tab10(np.linspace(0, 1, 10)), strict=False))
    for channel, frame in agg.groupby("channel", sort=False):
        reps_values = sorted(frame["reps"].unique())
        fig, axes = plt.subplots(
            2, len(reps_values), figsize=(4.2 * len(reps_values), 7), sharex=True, squeeze=False
        )
        for col, reps in enumerate(reps_values):
            sub = frame[frame["reps"] == reps]
            first = sub.drop_duplicates("p")
            top, bottom = axes[0][col], axes[1][col]
            top.loglog(
                first["p"], first["unmitigated_bias"].abs(), "k-o", lw=2, label="unmitigated"
            )
            bottom.loglog(first["p"], first["unmitigated_rmse"], "k-o", lw=2)
            for method, m in sub.groupby("method", sort=False):
                style = {"color": colours[str(method)], "marker": "s", "ls": "--"}
                top.loglog(m["p"], m["zne_bias"].abs().clip(lower=1e-5), label=str(method), **style)
                bottom.loglog(m["p"], m["zne_rmse"], **style)
            top.set_title(f"p = {reps!s}", fontsize=10)
            bottom.set_xlabel("p_e")
            for ax in (top, bottom):
                ax.grid(alpha=0.3, which="both")
        axes[0][0].set_ylabel("|energy bias|, fixed parameters")
        axes[1][0].set_ylabel(f"rmse at {int(points['shots'].iloc[0])} shots per basis and scale")
        axes[0][-1].legend(fontsize=7)
        fig.suptitle(f"ZNE under paper-layer {channel} noise (exact density matrices)", fontsize=10)
        fig.tight_layout()
        path = out_dir / f"zne_bias_{channel}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    first_source = levels["source"].iloc[0]
    sub = levels[levels["source"] == first_source]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for (channel, p, scaling), frame in sub.groupby(["channel", "p", "noise_scaling"], sort=False):
        if scaling == "none":
            continue
        base = sub[(sub["channel"] == channel) & (sub["p"] == p) & (sub["scale"] == 1)]
        pts = pd.concat([base, frame]).sort_values("scale")
        ax.plot(pts["scale"], pts["energy"], marker="o", label=f"{channel} {p:g} {scaling}")
    ax.axhline(float(sub["ideal_energy"].iloc[0]), color="k", lw=1, ls=":", label="noise-free")
    ax.set_xlabel("noise scale lambda")
    ax.set_ylabel("exact noisy energy")
    ax.set_title(first_source, fontsize=7)
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "zne_levels.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths


def summarise_zne_analysis(out_dir: Path) -> Path:
    """(Re)write ``report.md`` and figures from ``levels.csv`` / ``points.csv``."""
    out_dir = Path(out_dir)
    levels = pd.read_csv(out_dir / "levels.csv")
    points = pd.read_csv(out_dir / "points.csv")
    sampled_path = out_dir / "sampled.csv"
    sampled = pd.read_csv(sampled_path) if sampled_path.exists() else None
    title = out_dir.name
    analysis = out_dir / "analysis.toml"
    if analysis.exists():
        title = str(load_toml(analysis).get("name", title))
    (out_dir / "report.md").write_text(_report(points, sampled, title))
    _plot(levels, points, out_dir / "figures")
    return out_dir / "report.md"
