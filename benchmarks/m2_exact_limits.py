"""Infinite-shot limits of every M2 mitigation stack for graph_hva on PRB 106 layer noise.

Usage::

    uv run python benchmarks/m2_exact_limits.py configs/mitigation/graph_hva_m2_fixed.toml \
        [--probabilities 1e-3 3e-3 1e-2] [--readout-error 0.02] \
        [--sampled results/<ts>-graph_hva_m2_fixed results/<ts>-graph_hva_m2_fixed_strong]
    # fill the density-matrix cache in parallel first (one process per point and p):
    uv run python benchmarks/m2_exact_limits.py CONFIG --fill-only 1e-2 --points opt
    # only re-compare saved limits with sampled studies:
    uv run python benchmarks/m2_exact_limits.py CONFIG \
        --from-limits results/<ts>-graph_hva_m2_exact --sampled ...

The sampled M2 table (``kagome-vqe mitigation-study``) measures each stack 16 times with 1024
shots per basis. This script computes, from exact density matrices, what each stack converges
to with infinitely many shots, and the standard error a 1024-shot estimate should have, so that

* a bias can be attributed to the method (the limit) rather than to sampling;
* a cancellation between error sources is visible (main's M2 found one: filtering before
  readout correction over-corrects by ``1/(1 - 2q)^2`` and that happened to cancel the gate
  residual on FakeGuadalupeV2);
* every sampled cell can be checked against its limit (``z``) and its predicted spread.

For every point of the study config (and the twin), every gate-noise probability (alone,
``gates``, and with the readout error, ``full``) and readout error alone (``readout``):

* exact density matrices at ZNE probability scales 1, 2, 3 and folding scales 3, 5;
* per measurement basis, the outcome distribution; readout error applied analytically
  (independent symmetric flips ``q``) and corrected with the exact inverse;
* the estimators of ``LogicalNoiseEnergy`` (plain mean, readout correction, symmetry filter,
  correct-then-filter) with their per-shot variance by the same delta method, plus
  **filter-then-correct** (the order main's M2 withdrew), which is not implemented as a stack;
* ZNE by the same weighted exponential fit as ``ZneEnergy``, and twin rescaling with the
  twin's 16384-shot measurement noise in the predicted spread.

Not modelled: the finite readout calibration (2 x 16384 shots: a setup-to-setup spread of
about 0.1 % of the energy) and its re-calibration per ZNE scale.

Outputs ``results/<UTC>-graph_hva_m2_exact/``: ``limits.csv`` (one row per point, noise, stack),
``damping.csv`` (``E_noisy / E_ideal`` of target and twin per counts estimator) and
``report.md``.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kagome_vqe.mitigation.counts import _apply_inverse
from kagome_vqe.mitigation.wrappers import extrapolate
from kagome_vqe.mitigation_study import _theta, load_mitigation_study_config
from kagome_vqe.noise import (
    _BASIS_CHANGE,
    PauliTrajectorySimulator,
    exact_noisy_state,
    fold_noise_layers,
    layer_noise_circuit,
    measurement_bases,
    scale_noise_probability,
)
from kagome_vqe.vqe import prepare

log = logging.getLogger("m2_exact")

SHOTS = 1024
TWIN_SHOTS = 16384
PROB_SCALES = (1.0, 2.0, 3.0)
FOLD_SCALES = (1.0, 3.0, 5.0)
_LIMIT_COLUMNS = (
    "point",
    "noise",
    "stack",
    "limit",
    "ideal_energy",
    "limit_bias_percent",
    "predicted_se",
    "kept_weight",
)
COUNTS = ("none", "readout", "symmetry", "readout+symmetry", "filter_then_correct")


def basis_probabilities(rho: np.ndarray, basis: str) -> np.ndarray:
    """Diagonal of ``V rho V^dag`` with ``V`` the product of per-qubit basis changes."""
    n = len(basis)
    tensor = rho.reshape([2] * (2 * n))  # row axes 0..n-1, column axes n..2n-1; qubit q at n-1-q
    for q, letter in enumerate(basis):
        if letter == "Z":
            continue
        v = _BASIS_CHANGE[letter]
        row, col = n - 1 - q, 2 * n - 1 - q
        tensor = np.moveaxis(np.tensordot(v, tensor, axes=([1], [row])), 0, row)
        tensor = np.moveaxis(np.tensordot(v.conj(), tensor, axes=([1], [col])), 0, col)
    dim = 2**n
    return np.asarray(np.real(np.diagonal(tensor.reshape(dim, dim))), dtype=float)


def readout_matrices(n: int, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-qubit confusion matrices (columns: prepared bit) and their inverses."""
    a = np.array([[1 - q, q], [q, 1 - q]])
    return np.repeat(a[None], n, axis=0), np.repeat(np.linalg.inv(a)[None], n, axis=0)


def basis_statistic(
    probs: np.ndarray,
    diagonal: np.ndarray,
    *,
    symmetry: bool,
    inverses: np.ndarray | None,
    n: int,
) -> tuple[float, float, float]:
    """Infinite-shot value, one-shot variance and kept weight of one basis.

    The same estimator as ``LogicalNoiseEnergy._basis_statistics``: ``u = A^-T (mask d)``,
    ``v = A^-T mask``, estimate ``E[u] / E[v]`` under the read distribution ``probs``.
    """
    dim = probs.size
    mask = np.ones(dim)
    if symmetry:
        weight = np.bitwise_count(np.arange(dim, dtype=np.int64))
        mask = (weight == n // 2).astype(float)
    u, v = diagonal * mask, mask
    if inverses is not None:
        transposed = np.transpose(inverses, (0, 2, 1))
        u, v = _apply_inverse(u, transposed), _apply_inverse(v, transposed)
    u_mean, v_mean = float(probs @ u), float(probs @ v)
    if not symmetry and inverses is None:
        return u_mean, float(probs @ u**2) - u_mean**2, 1.0
    ratio = u_mean / v_mean
    var_u = float(probs @ u**2) - u_mean**2
    var_v = float(probs @ v**2) - v_mean**2
    cov = float(probs @ (u * v)) - u_mean * v_mean
    return ratio, (var_u - 2 * ratio * cov + ratio**2 * var_v) / v_mean**2, v_mean


def counts_estimates(
    probs: list[np.ndarray], diagonals: list[np.ndarray], q: float, n: int
) -> dict[str, tuple[float, float, float]]:
    """Every counts-level estimator: ``label -> (energy, one-shot variance, mean kept weight)``.

    ``probs`` are the pre-readout outcome distributions of the bases.
    """
    noise, inverse = readout_matrices(n, q)
    read = [_apply_inverse(p, noise) if q > 0 else p for p in probs]
    out: dict[str, tuple[float, float, float]] = {}
    for label, symmetry, correct in (
        ("none", False, False),
        ("readout", False, True),
        ("symmetry", True, False),
        ("readout+symmetry", True, True),
    ):
        stats = [
            basis_statistic(
                p, d, symmetry=symmetry, inverses=inverse if correct and q > 0 else None, n=n
            )
            for p, d in zip(read, diagonals, strict=True)
        ]
        out[label] = (
            sum(s[0] for s in stats),
            sum(s[1] for s in stats),
            float(np.mean([s[2] for s in stats])),
        )
    # Filter, then correct (main's withdrawn order): symmetric flips give v_q = s / (1 - 2q), so
    # every two-body term of the filtered read distribution is divided by (1 - 2q)^2.
    energy, variance, kept = out["symmetry"]
    factor = (1 - 2 * q) ** -2
    out["filter_then_correct"] = (energy * factor, variance * factor**2, kept)
    return out


def _zne(levels: dict[float, dict[str, Any]], scales: tuple[float, ...], inner: str) -> Any:
    energies = np.array([levels[s][inner][0] for s in scales])
    ses = np.sqrt([levels[s][inner][1] / SHOTS for s in scales])
    value, _ = extrapolate(scales, energies, ses, "exponential")
    # Predicted standard error by linear propagation through the fit (finite differences).
    grads = []
    for i in range(len(scales)):
        bumped = energies.copy()
        bumped[i] += 1e-6
        grads.append((extrapolate(scales, bumped, ses, "exponential")[0] - value) / 1e-6)
    se = float(np.sqrt(np.sum((np.asarray(grads) * ses) ** 2)))
    return value, se, energies


def analyse_point(
    circuit_for: Any,
    label: str,
    theta: np.ndarray,
    bases: list[tuple[str, np.ndarray]],
    settings: list[tuple[str, float, float]],
    n: int,
    cache_dir: Path,
) -> dict[str, dict[float, dict[float, dict[str, Any]]]]:
    """``levels[setting][scaling-key][scale] -> counts estimates``.

    The basis distributions of every (p, scaling, scale) are cached in ``cache_dir`` (one
    ``.npy`` each), so the density matrices can be computed by several processes and once.
    """
    letters = [b for b, _ in bases]
    diagonals = [d for _, d in bases]

    def distributions(p: float, scaling: str, scale: float) -> list[np.ndarray]:
        kind = scaling if scale != 1 else "none"
        path = cache_dir / f"{label}_p{p:g}_{kind}{scale:g}.npy"
        if not path.exists():
            tick = time.perf_counter()
            if p == 0:
                sim = PauliTrajectorySimulator(circuit_for(0.0))
                psi = sim.ideal_state(theta)
                rho = np.outer(psi, psi.conj())
            else:
                rho = exact_noisy_state(circuit_for(p, scaling, scale), theta)
            probs = np.vstack([basis_probabilities(rho, b) for b in letters])
            del rho
            np.save(path.with_suffix(".tmp.npy"), probs)
            path.with_suffix(".tmp.npy").rename(path)
            log.info(
                "%s rho p=%g %s x%g: %.1f s", label, p, scaling, scale, time.perf_counter() - tick
            )
        return list(np.load(path))

    out: dict[str, Any] = {}
    for setting, p, q in settings:
        prob = {
            s: counts_estimates(distributions(p, "probability", s), diagonals, q, n)
            for s in (PROB_SCALES if p > 0 else (1.0,))
        }
        fold = {
            s: counts_estimates(distributions(p, "folding", s), diagonals, q, n)
            for s in (FOLD_SCALES if p > 0 else (1.0,))
        }
        out[setting] = {"probability": prob, "folding": fold}
    return out


def stack_rows(  # noqa: D103 (rows of the limits table for one point and noise setting)
    point: str,
    setting: str,
    ideal: float,
    levels: dict[str, Any],
    twin_levels: dict[str, Any],
    twin_ideal: float,
) -> list[dict[str, Any]]:
    base = levels["probability"][1.0]
    twin_base = twin_levels["probability"][1.0]
    rows: list[dict[str, Any]] = []

    def add(stack: str, energy: float, se: float, kept: float = math.nan) -> None:
        rows.append(
            {
                "point": point,
                "noise": setting,
                "stack": stack,
                "limit": energy,
                "ideal_energy": ideal,
                "limit_bias_percent": 100 * (energy - ideal) / abs(ideal),
                "predicted_se": se,
                "kept_weight": kept,
            }
        )

    for label in COUNTS:
        energy, variance, kept = base[label]
        add(label, energy, math.sqrt(variance / SHOTS), kept)
    gated = len(levels["probability"]) > 1
    for stack, scaling, scales, inner in (
        ("zne_prob", "probability", PROB_SCALES, "none"),
        ("zne_fold", "folding", FOLD_SCALES, "none"),
        ("readout+zne_prob", "probability", PROB_SCALES, "readout"),
        ("readout+symmetry+zne_prob", "probability", PROB_SCALES, "readout+symmetry"),
    ):
        if not gated:  # no gate noise: every scale is the same circuit
            energy, variance, kept = base[inner]
            add(stack, energy, math.sqrt(variance / SHOTS), kept)
            continue
        value, se, _ = _zne(levels[scaling], scales, inner)
        add(stack, value, se)
    for stack, inner in (
        ("twin", "none"),
        ("readout+twin", "readout"),
        ("symmetry+twin", "symmetry"),
        ("readout+symmetry+twin", "readout+symmetry"),
    ):
        energy, variance, kept = base[inner]
        t_energy, t_variance, _ = twin_base[inner]
        factor = twin_ideal / t_energy
        twin_term = (energy * factor / t_energy) ** 2 * t_variance / TWIN_SHOTS
        se = math.sqrt(factor**2 * variance / SHOTS + twin_term)
        add(stack, energy * factor, se, kept)
    return rows


def _settings(probabilities: list[float], q: float) -> list[tuple[str, float, float]]:
    out: list[tuple[str, float, float]] = [("readout", 0.0, q)]
    for p in probabilities:
        suffix = "" if p == 1e-3 else f"_{p:g}"
        out += [(f"gates{suffix}", p, 0.0), (f"full{suffix}", p, q)]
    return out


def _compare(limits: pd.DataFrame, sampled_dirs: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(d / "points.csv") for d in sampled_dirs]
    if not frames:
        return limits
    sampled = pd.concat(frames, ignore_index=True)
    sampled = sampled[["point", "noise", "stack", "mean", "spread", "mean_reported_std", "repeats"]]
    merged = limits.merge(sampled, on=["point", "noise", "stack"], how="left")
    sem = merged["spread"] / np.sqrt(merged["repeats"])
    merged["sampled_bias_percent"] = (
        100 * (merged["mean"] - merged["ideal_energy"]) / merged["ideal_energy"].abs()
    )
    merged["z"] = (merged["mean"] - merged["limit"]) / sem
    merged["spread_over_predicted"] = merged["spread"] / merged["predicted_se"]
    merged["reported_over_spread"] = merged["mean_reported_std"] / merged["spread"]
    return merged


def _report(frame: pd.DataFrame, damping: pd.DataFrame) -> str:
    lines = [
        "# Infinite-shot limits of the M2 stacks (graph_hva, PRB 106 layer noise)",
        "",
        "`limit` = bias of the infinite-shot value in % of the noise-free energy; `sampled` = the",
        "16-repeat mean of the mitigation study; `z` = (sampled - limit) / SEM; `spread/pred` =",
        "observed spread between fresh setups over the predicted 1024-shot standard error.",
        "",
    ]
    for (point, noise), block in frame.groupby(["point", "noise"], sort=False):
        lines += [
            f"## {point}, {noise}",
            "",
            "| stack | limit % | sampled % | z | predicted se | spread/pred | kept |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, r in block.iterrows():
            sampled = r.get("sampled_bias_percent", math.nan)
            lines.append(
                f"| {r.stack} | {r.limit_bias_percent:+.2f} | "
                + ("" if pd.isna(sampled) else f"{sampled:+.2f}")
                + " | "
                + ("" if pd.isna(r.get("z", math.nan)) else f"{r.z:+.1f}")
                + f" | {r.predicted_se:.3f} | "
                + (
                    ""
                    if pd.isna(r.get("spread_over_predicted", math.nan))
                    else f"{r.spread_over_predicted:.2f}"
                )
                + " | "
                + ("" if pd.isna(r.kept_weight) else f"{r.kept_weight:.3f}")
                + " |"
            )
        lines.append("")
    lines += [
        "## Damping E_noisy / E_ideal per counts estimator (target points and twin)",
        "",
        damping.pivot_table(
            index=["noise", "estimator"], columns="point", values="damping", sort=False
        ).to_csv(float_format="%.4f"),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:  # noqa: PLR0915 (a linear script)
    """Compute and write the tables."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--probabilities", type=float, nargs="+", default=[1e-3, 3e-3, 1e-2])
    parser.add_argument("--readout-error", type=float, default=0.02)
    parser.add_argument("--sampled", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--from-limits", type=Path, default=None)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("results/m2_exact_cache"),
        help="Basis distributions per (point, p, scaling, scale).",
    )
    parser.add_argument(
        "--fill-only",
        type=float,
        nargs="*",
        default=None,
        help="Only compute (cache) these gate-noise probabilities, then stop.",
    )
    parser.add_argument("--points", nargs="*", default=None, help="With --fill-only.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.from_limits is not None:
        out = args.from_limits
        base = pd.read_csv(out / "limits.csv")
        keep = [c for c in base.columns if c in _LIMIT_COLUMNS]
        limits = _compare(base[keep], list(args.sampled))
        damping = pd.read_csv(out / "damping.csv")
        limits.to_csv(out / "limits_compared.csv", index=False)
        report = _report(limits, damping)
        (out / "report.md").write_text(report)
        sys.stdout.write(report)
        return
    cfg = load_mitigation_study_config(args.config)
    cells = cfg.cells()
    twin_cell = next(c for c in cells if c[2].mitigation.rescale is not None)
    prep = prepare(twin_cell[3])
    assert prep.twin is not None
    ansatz = prep.ansatz
    n = prep.problem.circuit.num_qubits
    bases = measurement_bases(prep.hamiltonian)

    def circuit_for(p: float, scaling: str = "none", scale: float = 1.0) -> Any:
        base = layer_noise_circuit(ansatz, "depolarizing", p)
        if scale == 1:
            return base
        if scaling == "probability":
            return scale_noise_probability(base, scale)
        return fold_noise_layers(base, (round(scale) - 1) // 2)

    settings = _settings(args.probabilities, args.readout_error)
    points = {
        pt.label: np.asarray(_theta(cfg, pt, ansatz.num_parameters), dtype=float)
        for pt in cfg.points
    }
    points = {
        k: (np.full(ansatz.num_parameters, v) if np.ndim(v) == 0 else v) for k, v in points.items()
    }
    points["twin"] = np.asarray(prep.twin[0], dtype=float)
    clean = PauliTrajectorySimulator(circuit_for(0.0))
    h = prep.hamiltonian.to_matrix(sparse=True)
    ideals = {}
    for k, theta in points.items():
        psi = clean.ideal_state(theta)
        ideals[k] = float(np.real(np.vdot(psi, h @ psi)))
    log.info("noise-free energies %s (twin reference %.6f)", ideals, prep.twin[1])
    cache_dir = args.cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    if args.fill_only is not None:  # fill the cache for some points / probabilities and stop
        wanted = [s for s in settings if s[1] in args.fill_only or not args.fill_only]
        for k in args.points or list(points):
            analyse_point(circuit_for, k, points[k], bases, wanted, n, cache_dir)
        return
    levels = {
        k: analyse_point(circuit_for, k, theta, bases, settings, n, cache_dir)
        for k, theta in points.items()
    }
    rows, damp = [], []
    for k in points:
        for setting, _, _ in settings:
            if k != "twin":
                rows += stack_rows(
                    k,
                    setting,
                    ideals[k],
                    levels[k][setting],
                    levels["twin"][setting],
                    ideals["twin"],
                )
            for est in COUNTS:
                energy = levels[k][setting]["probability"][1.0][est][0]
                damp.append(
                    {"point": k, "noise": setting, "estimator": est, "damping": energy / ideals[k]}
                )
    limits = _compare(pd.DataFrame(rows), list(args.sampled))
    damping = pd.DataFrame(damp)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.output_dir / f"{stamp}-graph_hva_m2_exact"
    out.mkdir(parents=True)
    limits.to_csv(out / "limits.csv", index=False)
    damping.to_csv(out / "damping.csv", index=False)
    report = _report(limits, damping)
    (out / "report.md").write_text(report)
    sys.stdout.write(report)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
