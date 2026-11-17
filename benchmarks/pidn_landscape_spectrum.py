"""Gaussian-smoothed contrast and low-frequency power of measured 2-D landscape slices.

Usage::

    uv run python benchmarks/pidn_landscape_spectrum.py measure DIR --pipelines raw rs \
        [--grid 24] [--pairs 0]
    uv run python benchmarks/pidn_landscape_spectrum.py report DIR

Liu and Wang (arXiv:2605.02066, Sec. IV A, Eqs. 39-42) characterise a landscape slice ``E(theta_1,
theta_2)`` (all other parameters fixed) by the contrast of its Gaussian-smoothed version
``E~ = G_sigma * E`` -- ``Delta E_smooth = max E~ - min E~`` and ``sigma_smooth = std E~`` -- and by
the low-frequency power ratio ``R_LF = sum_{k in K_low} |E^(k)|^2 / sum_k |E^(k)|^2``; they link
PIDN's success to a high ``R_LF`` (their Fig. 9). Gaussian smoothing is a *diagnostic* there,
not part of PIDN. This script measures the same quantities for graph_hva slices through the
point where PIDN takes over (``theta_10`` of the real readout+symmetry gradient-descent run,
seed 1), for every measurement pipeline.

Conventions (the paper gives neither ``sigma`` nor ``K_low``): each exchange angle enters the
energy with period ``pi``, so a slice is sampled on a periodic ``grid x grid`` lattice over one
period around ``theta_10``; the mean is removed before the spectrum (otherwise the constant term
dominates ``R_LF``); ``K_low`` is the support of the *noise-free* slice's spectrum (the noise-free
energy is a trigonometric polynomial of low degree, so its ``R_LF`` is 1 by construction);
``G_sigma`` is a periodic Gaussian of one grid spacing. Pipelines ``raw``, ``rs`` (readout +
symmetry), ``rzne`` (readout + probability ZNE 1, 2, 3) and ``rs_x8`` (readout + symmetry at 8192
shots per basis) are measured with the trajectory sampler under depolarising ``p_e = 1e-3`` +
readout error 0.02; ``ideal`` is the statevector; ``exact_noise`` is the noise-free slice times
the readout+symmetry damping (what infinitely many shots would give); ``rs_x64`` and
``rs_x512`` add Gaussian noise of the measured readout+symmetry spread scaled by
``1/sqrt(k)``.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("landscape")

SOURCE = "results/*-graph_hva_m2_gd/runs/*-full_rs-gd_ps60-s1"
CENTRE_ITERATION = 10
PAIRS_SEED = 2026
NUM_PAIRS = 3
PIPELINES: dict[str, dict[str, Any]] = {
    "raw": {"shots": 1024, "mitigation": {}},
    "rs": {"shots": 1024, "mitigation": {"readout": True, "symmetry": True}},
    "rzne": {
        "shots": 1024,
        "mitigation": {
            "readout": True,
            "zne": {
                "scales": [1, 2, 3],
                "extrapolation": "exponential",
                "noise_scaling": "probability",
            },
        },
    },
    "rs_x8": {"shots": 8192, "mitigation": {"readout": True, "symmetry": True}},
}


def _setup() -> tuple[Any, np.ndarray, list[tuple[int, int]]]:
    from kagome_vqe.results import load_run

    source = Path(sorted(glob.glob(SOURCE))[0])
    run = load_run(source)
    assert run.trajectory is not None
    centre = run.trajectory["theta"][CENTRE_ITERATION]
    rng = np.random.default_rng(PAIRS_SEED)
    pairs = [
        tuple(int(i) for i in rng.choice(centre.size, 2, replace=False)) for _ in range(NUM_PAIRS)
    ]
    return run, centre, pairs  # type: ignore[return-value]


def grid_points(centre: np.ndarray, pair: tuple[int, int], grid: int) -> np.ndarray:
    """Parameter vectors of a periodic ``grid x grid`` slice over one period (pi)."""
    offsets = (np.arange(grid) - grid // 2) * math.pi / grid
    points = np.repeat(centre[None, :], grid * grid, axis=0)
    a, b = np.meshgrid(offsets, offsets, indexing="ij")
    points[:, pair[0]] += a.ravel()
    points[:, pair[1]] += b.ravel()
    return points


def measure(args: argparse.Namespace) -> None:
    """Measure the slices of the requested pipelines (``ideal`` always)."""
    from kagome_vqe.config import BackendConfig, MitigationConfig
    from kagome_vqe.estimators import make_estimator
    from kagome_vqe.vqe import _ideal_energy_fn, prepare

    run, centre, pairs = _setup()
    out = args.dir
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": str(run.path),
        "centre_iteration": CENTRE_ITERATION,
        "pairs": pairs,
        "grid": args.grid,
    }
    (out / "slices.json").write_text(json.dumps(meta, indent=2) + "\n")
    prep = prepare(run.config)
    mapping = prep.parameter_map
    ideal_fn = _ideal_energy_fn(prep)
    for p, pair in enumerate(pairs):
        if args.pairs is not None and p not in args.pairs:
            continue
        points = grid_points(centre, pair, args.grid)
        path = out / f"pair{p}_ideal.npy"
        if not path.exists():
            np.save(path, np.array([ideal_fn(x) for x in points]).reshape(args.grid, args.grid))
        for name in args.pipelines:
            path = out / f"pair{p}_{name}.npy"
            if path.exists():
                continue
            spec = PIPELINES[name]
            backend = BackendConfig.model_validate(
                run.config.backend.model_dump(mode="json") | {"shots": spec["shots"]}
            )
            mitigation = MitigationConfig.model_validate(spec["mitigation"])
            tick = time.perf_counter()
            with make_estimator(
                backend, prep.problem, None, seed=1000 * p + len(name), mitigation=mitigation
            ) as est:
                energies = est.evaluate(np.vstack([mapping.expand(x) for x in points])).energies
            np.save(path, np.asarray(energies).reshape(args.grid, args.grid))
            log.info("pair %d %s: %.0f s", p, name, time.perf_counter() - tick)


def _smooth(e: np.ndarray, sigma: float) -> np.ndarray:
    """Periodic Gaussian smoothing (FFT)."""
    k = np.fft.fftfreq(e.shape[0]) * e.shape[0]
    kernel = np.exp(
        -0.5 * (2 * math.pi * sigma / e.shape[0]) ** 2 * (k[:, None] ** 2 + k[None, :] ** 2)
    )
    return np.real(np.fft.ifft2(np.fft.fft2(e) * kernel))


def metrics(e: np.ndarray, low: np.ndarray, sigma: float = 1.0) -> dict[str, float]:
    """``Delta E_smooth``, ``sigma_smooth`` and ``R_LF`` (mean removed) of one slice."""
    smoothed = _smooth(e, sigma)
    power = np.abs(np.fft.fft2(e - e.mean())) ** 2
    return {
        "delta_e_smooth": float(smoothed.max() - smoothed.min()),
        "sigma_smooth": float(smoothed.std()),
        "r_lf": float(power[low].sum() / power.sum()),
    }


def report(args: argparse.Namespace) -> None:
    """Metrics of every measured slice, plus the exact-noise and emulated-shot references."""
    meta = json.loads((args.dir / "slices.json").read_text())
    rng = np.random.default_rng(7)
    rows = []
    for p in range(len(meta["pairs"])):
        ideal = np.load(args.dir / f"pair{p}_ideal.npy")
        spectrum = np.abs(np.fft.fft2(ideal - ideal.mean())) ** 2
        low = spectrum > 1e-10 * spectrum.max()
        slices = {"ideal": ideal}
        for name in PIPELINES:
            path = args.dir / f"pair{p}_{name}.npy"
            if path.exists():
                slices[name] = np.load(path)
        if "rs" in slices:
            residual = slices["rs"] - ideal * np.sum(slices["rs"] * ideal) / np.sum(ideal**2)
            damping = float(np.sum(slices["rs"] * ideal) / np.sum(ideal**2))
            slices["exact_noise"] = damping * ideal
            for k in (64, 512):
                slices[f"rs_x{k}"] = damping * ideal + residual.std() / math.sqrt(
                    k
                ) * rng.standard_normal(ideal.shape)
        for name, e in slices.items():
            noise = e - ideal * np.sum(e * ideal) / np.sum(ideal**2)
            rows.append(
                {"pair": p, "params": str(meta["pairs"][p]), "pipeline": name}
                | metrics(e, low)
                | {
                    "noise_std": float(noise.std()),
                    "signal_std": float(ideal.std()),
                    "low_modes": int(low.sum()),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.dir / "metrics.csv", index=False)
    mean = frame.groupby("pipeline", sort=False).mean(numeric_only=True)
    lines = [
        f"# Landscape smoothness (Liu and Wang Eqs. 39-42): {args.dir.name}",
        "",
        f"Slices through theta_{meta['centre_iteration']} of {Path(meta['source']).name}, "
        f"parameter pairs {meta['pairs']}, {meta['grid']}x{meta['grid']} grid over one period; "
        "means over pairs.",
        "",
        "| pipeline | R_LF | Delta E_smooth | sigma_smooth | noise std per point "
        "| noise-free std |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in mean.iterrows():
        lines.append(
            f"| {name} | {r.r_lf:.3f} | {r.delta_e_smooth:.3f} | {r.sigma_smooth:.3f} | "
            f"{r.noise_std:.3f} | {r.signal_std:.3f} |"
        )
    text = "\n".join(lines) + "\n"
    (args.dir / "report.md").write_text(text)
    sys.stdout.write(text)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure")
    m.add_argument("dir", type=Path)
    m.add_argument("--pipelines", nargs="*", default=list(PIPELINES))
    m.add_argument("--grid", type=int, default=24)
    m.add_argument("--pairs", type=int, nargs="*", default=None)
    r = sub.add_parser("report")
    r.add_argument("dir", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    {"measure": measure, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
