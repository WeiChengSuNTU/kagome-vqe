"""PIDN against Stage I signal-to-noise: shots, network capacity and Gaussian smoothing.

Usage::

    # 1. emulated Stage I / gradient-descent runs (one process per label is fine)
    uv run python benchmarks/pidn_shot_scaling.py generate [--out DIR] [--only k8 ...]
    # 2. open-loop tests: capacity sweep, overfitting diagnostic, smoothed targets
    uv run python benchmarks/pidn_shot_scaling.py openloop DIR \
        configs/pidn/graph_hva_pidn_capacity.toml \
        [--only k8 ...]
    # 3. tables
    uv run python benchmarks/pidn_shot_scaling.py report DIR
    # 4. closed loop, then its table against the emulated gradient descent
    uv run kagome-vqe pidn configs/pidn/graph_hva_pidn_shot_scaling.toml
    uv run python benchmarks/pidn_shot_scaling.py closed DIR \
        results/<ts>-graph_hva_pidn_shot_scaling_closed

Questions: (i) can more shots alone make Stage I learnable, and how many? (ii) is the network
too large for ten training points (does it memorise the shot noise)? (iii) does Gaussian
smoothing (Liu and Wang, Eq. 39) help?

Real readout+symmetry runs at 8192 shots per basis take ~25 min for 12 iterations, so larger
budgets are **emulated**: parameter-shift gradient descent on the exact noise-free gradient
plus independent Gaussian noise,

    g_t = c g_ideal(theta_t) + sigma / sqrt(k) xi_t,     value_t = c_v E_ideal(theta_t) + ...,

with ``c`` (the remaining damping) and the per-component ``sigma`` calibrated on the real
1024-shot readout+symmetry runs of ``graph_hva_m2_gd`` (``k`` = shots / 1024). The components
are independent because every parameter-shift pair uses its own shots. The calibration is
checked against the real 8192-shot runs (``sigma_8 / sigma_1`` against ``1/sqrt(8)``), and the
open-loop numbers of emulated ``k = 1, 8`` against the real paths. PIDN's noisy input
``C_noisy(theta_t)`` is *measured* (real readout+symmetry energy, 1024 shots per basis), as in
Stage III. ``ema`` labels smooth the Stage I *path* instead: the step uses an exponential moving
average of the gradients (bias-corrected, ``beta``); the recorded targets stay the raw ones.

Each run directory has ``config.toml`` (the real run's, 60 iterations), ``trajectory.npz``,
``result.json`` (``pipeline`` = label, ``emulated`` = calibration and settings), so
``kagome-vqe pidn`` and the open-loop test accept it as a source.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tomli_w

from kagome_vqe.pidn.diagnostics import load_trajectory, target_quality
from kagome_vqe.pidn.model import build_batch
from kagome_vqe.pidn.training import PidnEnsemble
from kagome_vqe.pidn.workflow import PidnVariant, _surrogate, load_pidn_config
from kagome_vqe.trajectory_analysis import cosine

log = logging.getLogger("pidn_shot_scaling")

REAL_1X = "results/*-graph_hva_m2_gd/runs/*-full_rs-gd_ps60-s[123]"
REAL_8X = "results/*-graph_hva_m2_gd/runs/*-full_rs_x8-gd_ps12-s[123]"
PLAN: list[tuple[str, float, float]] = [
    ("k1", 1, 0.0),
    ("k8", 8, 0.0),
    ("k64", 64, 0.0),
    ("k512", 512, 0.0),
    ("k4096", 4096, 0.0),
    ("kinf", math.inf, 0.0),
    ("k1_ema0.9", 1, 0.9),
    ("k8_ema0.9", 8, 0.9),
]
T_INIT = 10
ITERATIONS = 60


# -- calibration --------------------------------------------------------------------------------


def calibrate(real: list[Path], real8: list[Path]) -> dict[str, Any]:
    """Damping and per-component noise of the real readout+symmetry gradients."""
    trajs = [load_trajectory(p) for p in real]
    g = np.vstack([t["grad"] for t in trajs])
    gi = np.vstack([t["ideal_grad"] for t in trajs])
    v = np.concatenate([t["value"] for t in trajs])
    vi = np.concatenate([t["ideal_energy"] for t in trajs])
    c = float(np.sum(g * gi) / np.sum(gi * gi))
    c_v = float(np.sum(v * vi) / np.sum(vi * vi))
    sigma = np.sqrt(np.mean((g - c * gi) ** 2, axis=0))
    sigma_v = float(np.std(v - c_v * vi))
    out: dict[str, Any] = {
        "sources": [str(p) for p in real],
        "c": c,
        "c_value": c_v,
        "sigma": sigma.tolist(),
        "sigma_norm": float(np.linalg.norm(sigma)),
        "sigma_value": sigma_v,
    }
    if real8:
        t8 = [load_trajectory(p) for p in real8]
        g8 = np.vstack([t["grad"] for t in t8])
        gi8 = np.vstack([t["ideal_grad"] for t in t8])
        sigma8 = np.sqrt(np.mean((g8 - c * gi8) ** 2, axis=0))
        out["sigma8_over_sigma1"] = float(np.linalg.norm(sigma8) / np.linalg.norm(sigma))
        out["expected_ratio"] = 1 / math.sqrt(8)
        out["sources_8x"] = [str(p) for p in real8]
    return out


# -- generation ---------------------------------------------------------------------------------


class _Problem:
    """The real run's problem: noise-free energy and gradient, measured noisy energies."""

    def __init__(self, source: Path, seed: int) -> None:
        from kagome_vqe.estimators import make_estimator
        from kagome_vqe.results import load_run
        from kagome_vqe.vqe import _ideal_energy_fn, _ideal_gradient_fn, prepare

        self.run = load_run(source)
        cfg = self.run.config
        self.prep = prepare(cfg)
        self.map = self.prep.parameter_map
        tracked = replace(self.prep, config=cfg.model_copy(update={"track_ideal_gradient": True}))
        self.ideal_gradient = _ideal_gradient_fn(tracked)
        self.ideal_energy = _ideal_energy_fn(self.prep)
        self._noisy = make_estimator(
            cfg.backend, self.prep.problem, None, seed=seed, mitigation=cfg.mitigation
        )
        self._noisy.__enter__()
        self.groups = self.prep.problem.measurement_groups
        self.shots = int(cfg.backend.shots or 0)

    def noisy(self, theta: np.ndarray) -> float:
        return float(self._noisy.evaluate(self.map.expand(theta)[None, :]).energies[0])

    def close(self) -> None:
        self._noisy.__exit__(None, None, None)


def emulate(
    problem: _Problem, calib: dict[str, Any], *, k: float, beta: float, seed: int
) -> dict[str, np.ndarray]:
    """One emulated gradient-descent run of ``ITERATIONS`` steps from the real run's start."""
    real = problem.run.trajectory
    assert real is not None
    lr = float(real["learning_rate"])
    n = real["theta"].shape[1]
    rng = np.random.default_rng(seed + 1000)  # the same noise draws for every k (common numbers)
    scale = 0.0 if math.isinf(k) else 1 / math.sqrt(k)
    sigma = np.asarray(calib["sigma"]) * scale
    sigma_v = calib["sigma_value"] * scale
    theta = real["theta"][0].copy()
    rows: dict[str, list[Any]] = {
        key: []
        for key in ("theta", "theta_next", "grad", "value", "noisy", "ideal_energy", "ideal_grad")
    }
    momentum = np.zeros(n)
    for t in range(ITERATIONS):
        gi = np.asarray(problem.ideal_gradient(theta), dtype=float)
        ei = float(problem.ideal_energy(theta))
        xi, eta = rng.standard_normal(n), rng.standard_normal()
        grad = calib["c"] * gi + sigma * xi
        value = calib["c_value"] * ei + sigma_v * eta
        if beta > 0:
            momentum = beta * momentum + (1 - beta) * grad
            step = momentum / (1 - beta ** (t + 1))
        else:
            step = grad
        nxt = theta - lr * step
        for key, val in (
            ("theta", theta.copy()),
            ("theta_next", nxt.copy()),
            ("grad", grad),
            ("value", value),
            ("noisy", problem.noisy(theta)),
            ("ideal_energy", ei),
            ("ideal_grad", gi),
        ):
            rows[key].append(val)
        theta = nxt
    energies = 2 * n + 1
    evaluations = energies * np.arange(1, ITERATIONS + 1, dtype=np.int64)
    executions = evaluations * problem.groups
    per_basis = 0 if math.isinf(k) else round(problem.shots * k)
    value = np.asarray(rows["value"])
    grad = np.vstack(rows["grad"])
    return {
        "theta": np.vstack(rows["theta"]),
        "theta_next": np.vstack(rows["theta_next"]),
        "grad": grad,
        "value": value,
        "value_std": np.full(ITERATIONS, sigma_v),
        "scales": np.ones(1),
        "scale_values": np.asarray(rows["noisy"])[:, None],
        "scale_stds": np.full((ITERATIONS, 1), np.nan),
        "scale_grads": grad[:, None, :],
        "zne_weights": np.ones((ITERATIONS, 1)),
        "learning_rate": np.asarray(lr),
        "evaluations": evaluations,
        "circuit_executions": executions,
        "shots": executions * per_basis,
        "ideal_energy": np.asarray(rows["ideal_energy"]),
        "ideal_grad": np.vstack(rows["ideal_grad"]),
    }


def write_run(
    out: Path, problem: _Problem, traj: dict[str, np.ndarray], meta: dict[str, Any]
) -> None:
    """A run directory the PIDN workflow accepts as a source."""
    from kagome_vqe.results import environment_info

    out.mkdir(parents=True, exist_ok=False)
    cfg = problem.run.config
    data = cfg.model_dump(mode="json")
    data["name"] = f"emulated-{meta['pipeline']}-s{cfg.seed}"
    data["optimizer"] = {**data["optimizer"], "maxiter": ITERATIONS}
    with (out / "config.toml").open("wb") as fh:
        tomli_w.dump(type(cfg).model_validate(data).to_toml_dict(), fh)
    np.savez(out / "trajectory.npz", **traj)  # type: ignore[arg-type]
    pd.DataFrame(
        {
            "iteration": np.arange(ITERATIONS),
            "value": traj["value"],
            "noisy": traj["scale_values"][:, 0],
        }
    ).to_csv(out / "trace.csv", index=False)
    exact = float(problem.run.result["exact_energy"])
    final = float(problem.ideal_energy(traj["theta_next"][-1]))
    result = {
        "schema": 2,
        "name": data["name"],
        "seed": cfg.seed,
        "pipeline": meta["pipeline"],
        "emulated": meta,
        "exact_energy": exact,
        "final_energy": math.nan,
        "ideal_energy": final,
        "ideal_relative_error": abs(final - exact) / abs(exact),
        "optimal_params": [float(x) for x in traj["theta_next"][-1]],
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    (out / "environment.json").write_text(json.dumps(environment_info(), indent=2) + "\n")
    (out / "status.json").write_text(json.dumps({"status": "completed"}) + "\n")


def generate(args: argparse.Namespace) -> None:
    """Calibrate on the real runs and write the emulated runs."""
    real = [Path(p) for p in sorted(glob.glob(REAL_1X))]
    real8 = [Path(p) for p in sorted(glob.glob(REAL_8X))]
    if not real:
        msg = f"no real runs match {REAL_1X}"
        raise FileNotFoundError(msg)
    out = args.out
    if out is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = Path("results") / f"{stamp}-graph_hva_pidn_shot_scaling"
    (out / "runs").mkdir(parents=True, exist_ok=True)
    calib_path = out / "calibration.json"
    if calib_path.exists():
        calib = json.loads(calib_path.read_text())
    else:
        calib = calibrate(real, real8)
        calib_path.write_text(json.dumps(calib, indent=2) + "\n")
    log.info(
        "calibration: c %.4f, |sigma| %.3f, sigma_8/sigma_1 %.3f (1/sqrt 8 = %.3f)",
        calib["c"],
        calib["sigma_norm"],
        calib.get("sigma8_over_sigma1", math.nan),
        1 / math.sqrt(8),
    )
    for label, k, beta in PLAN:
        if args.only and label not in args.only:
            continue
        for source in real:
            tick = time.perf_counter()
            seed = int(source.name.rsplit("-s", 1)[1])
            target = out / "runs" / f"{label}-s{seed}"
            if (target / "status.json").exists():
                continue
            problem = _Problem(source, seed + 31_337)
            try:
                traj = emulate(problem, calib, k=k, beta=beta, seed=seed)
                meta = {
                    "pipeline": label,
                    "shots_factor": k,
                    "ema_beta": beta,
                    "source": str(source),
                    "calibration": str(calib_path),
                }
                write_run(target, problem, traj, meta)
            finally:
                problem.close()
            err = abs(traj["ideal_energy"][-1] - problem.run.result["exact_energy"]) / abs(
                problem.run.result["exact_energy"]
            )
            log.info("%s: %.0f s, last error %.4f", target.name, time.perf_counter() - tick, err)
    sys.stdout.write(f"{out}\n")


# -- open loop ----------------------------------------------------------------------------------


def smooth_targets(traj: dict[str, np.ndarray], t_init: int) -> tuple[dict[str, np.ndarray], float]:
    """Gaussian-kernel smoothing of the Stage I targets in parameter space.

    ``g_hat(theta_t) = sum_s w_ts g_s / sum_s w_ts`` with ``w_ts = exp(-|theta_t - theta_s|^2 /
    2 h^2)`` over the Stage I points (the same for the value). The width ``h`` is chosen by
    leave-one-out cross-validation on the *measured* targets, so no noise-free information is
    used. Returns the trajectory with smoothed Stage I targets and ``h``.
    """
    theta = traj["theta"][:t_init]
    grads = traj["grad"][:t_init]
    dist2 = np.sum((theta[:, None, :] - theta[None, :, :]) ** 2, axis=-1)
    steps = np.linalg.norm(traj["theta_next"][:t_init] - theta, axis=1)
    best, best_h = math.inf, float(np.median(steps))
    for h in np.median(steps) * np.array([0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8]):
        w = np.exp(-dist2 / (2 * h**2))
        np.fill_diagonal(w, 0.0)
        loo = (w @ grads) / w.sum(axis=1, keepdims=True)
        score = float(np.mean(np.sum((loo - grads) ** 2, axis=1)))
        if score < best:
            best, best_h = score, float(h)
    w = np.exp(-dist2 / (2 * best_h**2))
    w /= w.sum(axis=1, keepdims=True)
    out = {key: np.array(value, copy=True) for key, value in traj.items()}
    out["grad"][:t_init] = w @ grads
    out["value"][:t_init] = w @ traj["value"][:t_init]
    return out, best_h


def _parameter_count(surrogate: Any) -> int:
    if isinstance(surrogate, PidnEnsemble) and surrogate.members:
        return int(sum(t.value.size for t in surrogate.members[0].params.tensors.values()))
    return 0


def evaluate(
    traj: dict[str, np.ndarray], variant: PidnVariant, *, t_init: int, end: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Open-loop continuation (as ``extrapolation_test``) plus the fit on the training points."""
    theta = np.vstack([traj["theta"], traj["theta_next"][-1:]])
    noisy = traj["scale_values"][:, 0]
    values, grads = traj["value"], traj["grad"]
    steps = traj["theta_next"] - traj["theta"]
    ideal = traj["ideal_grad"]
    window = variant.model.history
    train = build_batch(
        theta[:t_init],
        noisy[:t_init],
        window=window,
        values=values[:t_init],
        gradients=grads[:t_init],
        steps=steps[:t_init],
    )
    surrogate = _surrogate(variant)
    tick = time.perf_counter()
    surrogate.fit(train)
    seconds = time.perf_counter() - tick
    fit_query = build_batch(theta[:t_init], noisy[:t_init], window=window, rows=list(range(t_init)))
    _, fitted, _ = surrogate.predict(fit_query)
    query = build_batch(theta[:end], noisy[:end], window=window, rows=list(range(t_init, end)))
    _, predicted, _ = surrogate.predict(query)
    cos_ideal = cosine(predicted, ideal[t_init:end])
    rows = [
        {
            "variant": variant.label,
            "step": t,
            "ahead": t - t_init + 1,
            "cosine_to_ideal": float(cos_ideal[i]),
        }
        for i, t in enumerate(range(t_init, end))
    ]
    fit = {
        "variant": variant.label,
        "weights": _parameter_count(surrogate),
        "train_seconds": seconds,
        # memorisation: a fit that reproduces the noisy targets (cos -> 1) while its cosine to the
        # noise-free gradient stays at the targets' own value has learned the noise
        "fit_cos_to_targets": float(np.mean(cosine(fitted, grads[:t_init]))),
        "fit_cos_to_ideal": float(np.mean(cosine(fitted, ideal[:t_init]))),
        "targets_cos_to_ideal": float(np.mean(cosine(grads[:t_init], ideal[:t_init]))),
    }
    return rows, fit


def path_smoothness(traj: dict[str, np.ndarray], t_init: int, end: int) -> dict[str, float]:
    """How much the noise-free gradient turns between consecutive iterates."""
    ideal = traj["ideal_grad"]
    consecutive = cosine(ideal[1:], ideal[:-1])
    return {
        "consecutive_cos_stage1": float(np.mean(consecutive[: t_init - 1])),
        "consecutive_cos_continuation": float(np.mean(consecutive[t_init - 1 : end - 1])),
        "ideal_grad_norm_stage1": float(np.mean(np.linalg.norm(ideal[:t_init], axis=1))),
    }


def openloop(args: argparse.Namespace) -> None:
    """Open-loop tests of every variant (and smoothed targets) on every run of ``DIR``."""
    cfg = load_pidn_config(args.config)
    smooth = set(args.smooth)
    sources: list[tuple[str, Path]] = []
    for path in sorted((args.dir / "runs").glob("*")):
        label = path.name.rsplit("-s", 1)[0]
        if (path / "status.json").exists():
            sources.append((label, path))
    if args.real:
        sources += [("real_k1", Path(p)) for p in sorted(glob.glob(REAL_1X))]
        sources += [("real_k8", Path(p)) for p in sorted(glob.glob(REAL_8X))]
    if args.only:
        sources = [(lab, p) for lab, p in sources if lab in args.only]
    loops, fits, quality = [], [], []
    tag = "-".join(args.only) if args.only else "all"
    out = args.dir / "summary"
    out.mkdir(exist_ok=True)
    for label, path in sources:
        traj = load_trajectory(path)
        end = min(T_INIT + 20, len(traj["value"]))
        info = {"pipeline": label, "run": path.name}
        smoothed, h = smooth_targets(traj, T_INIT)
        quality.append(
            info
            | target_quality(traj, t_init=T_INIT)
            | path_smoothness(traj, T_INIT, end)
            | {
                "smoothed_cos_to_ideal": float(
                    np.mean(cosine(smoothed["grad"][:T_INIT], traj["ideal_grad"][:T_INIT]))
                ),
                "smoothing_width": h,
            }
        )
        for variant in cfg.variants:
            for targets, suffix in ((traj, ""), (smoothed, "+gsmooth")):
                if suffix and variant.label not in smooth:
                    continue
                v = variant.model_copy(update={"label": variant.label + suffix})
                rows, fit = evaluate(targets, v, t_init=T_INIT, end=end)
                loops += [info | r for r in rows]
                fits.append(info | fit)
                log.info(
                    "%s %s: open-loop cos %.3f (1-5), fit cos targets %.3f ideal %.3f",
                    path.name,
                    v.label,
                    np.mean([r["cosine_to_ideal"] for r in rows[:5]]),
                    fit["fit_cos_to_targets"],
                    fit["fit_cos_to_ideal"],
                )
        pd.DataFrame(loops).to_csv(out / f"open_loop_{tag}.csv", index=False)
        pd.DataFrame(fits).to_csv(out / f"fit_{tag}.csv", index=False)
        pd.DataFrame(quality).to_csv(out / f"targets_{tag}.csv", index=False)


# -- report -------------------------------------------------------------------------------------


def report(args: argparse.Namespace) -> None:
    """Merge the per-process CSVs and write ``summary/report.md``."""
    out = args.dir / "summary"
    loops = pd.concat([pd.read_csv(p) for p in sorted(out.glob("open_loop_*.csv"))])
    fits = pd.concat([pd.read_csv(p) for p in sorted(out.glob("fit_*.csv"))])
    quality = pd.concat([pd.read_csv(p) for p in sorted(out.glob("targets_*.csv"))])
    order = [p for p, _, _ in PLAN] + ["real_k1", "real_k8"]
    loops["window"] = np.where(loops["ahead"] <= 5, "1-5", "6-20")
    o = loops.pivot_table(
        index="pipeline", columns=["variant", "window"], values="cosine_to_ideal"
    ).reindex([p for p in order if p in set(loops["pipeline"])])
    q = quality.groupby("pipeline").mean(numeric_only=True).reindex(o.index)
    f = fits.groupby(["pipeline", "variant"]).mean(numeric_only=True)
    lines = [
        f"# PIDN against Stage I signal-to-noise: {args.dir.name}",
        "",
        "Means over seeds 1-3. `k` = shots per basis / 1024 (emulated; `real_*` = measured runs).",
        "",
        "## Stage I targets and path",
        "",
        "| pipeline | target cos | SNR late | Gaussian-smoothed target cos | consecutive cos "
        "(Stage I) | consecutive cos (continuation) | noise-free grad norm |",
        "|---|---|---|---|---|---|---|",
    ]
    for p, r in q.iterrows():
        lines.append(
            f"| {p} | {r.cosine_to_ideal:.3f} | {r.signal_to_noise_late:.2f} | "
            f"{r.smoothed_cos_to_ideal:.3f} | {r.consecutive_cos_stage1:.3f} | "
            f"{r.consecutive_cos_continuation:.3f} | {r.ideal_grad_norm_stage1:.3f} |"
        )
    header = [f"{v} {w}" for v, w in o.columns]
    lines += [
        "",
        "## Open-loop cosine to the noise-free gradient (steps ahead 1-5 / 6-20)",
        "",
        "| pipeline | " + " | ".join(header) + " |",
        "|---|" + "---|" * len(header),
    ]
    lines += [
        f"| {p} | " + " | ".join("" if pd.isna(v) else f"{v:.2f}" for v in row) + " |"
        for p, row in o.iterrows()
    ]
    lines += [
        "",
        "## Fit on the training points (memorisation test)",
        "",
        "`fit->targets`: cosine of the fitted gradient with the measured targets; `fit->ideal`:",
        "with the noise-free gradient; `targets->ideal`: the targets' own cosine. Memorising the",
        "noise means fit->targets near 1 and fit->ideal near targets->ideal.",
        "",
        "| pipeline | variant | weights | fit->targets | fit->ideal | targets->ideal |",
        "|---|---|---|---|---|---|",
    ]
    for (p, v), r in f.iterrows():
        lines.append(
            f"| {p} | {v} | {r.weights:.0f} | {r.fit_cos_to_targets:.3f} | "
            f"{r.fit_cos_to_ideal:.3f} | {r.targets_cos_to_ideal:.3f} |"
        )
    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text)
    sys.stdout.write(text)


def closed(args: argparse.Namespace) -> None:
    """Closed-loop PIDN (a ``kagome-vqe pidn`` study on the emulated runs) against emulated GD."""
    rows = []
    for path in sorted((args.dir / "runs").glob("*")):
        res = json.loads((path / "result.json").read_text())
        traj = load_trajectory(path)
        exact = res["exact_energy"]
        err = np.abs(traj["ideal_energy"] - exact) / abs(exact)
        rows.append(
            {
                "pipeline": res["pipeline"],
                "seed": res["seed"],
                "method": "GD",
                "error_t_init": float(err[T_INIT]),
                "final_error": float(res["ideal_relative_error"]),
                "best_error_stage3": float(err[T_INIT:].min()),
                "first_worse": math.nan,
                "stage3_cosine": 1.0,
                "shots": int(traj["shots"][-1]),
            }
        )
    for path in sorted((args.pidn / "runs").glob("*")):
        if not (path / "result.json").exists():
            continue
        res = json.loads((path / "result.json").read_text())
        traj = load_trajectory(path)
        exact = res["exact_energy"]
        err = np.abs(traj["ideal_energy"] - exact) / abs(exact)
        phase3 = traj["phase"] > 0
        worse = np.nonzero(phase3 & (err > err[T_INIT - 1]))[0]
        cos = cosine(traj["surrogate_grad"][phase3], traj["ideal_grad"][phase3])
        final_err = res["final_ideal_relative_error"]
        rows.append(
            {
                "pipeline": res["source_pipeline"],
                "seed": res["seed"],
                "method": res["variant"],
                "error_t_init": float(err[T_INIT - 1]),
                "final_error": float(final_err),
                "best_error_stage3": float(err[phase3].min()),
                "first_worse": float(worse[0]) if worse.size else math.nan,
                "stage3_cosine": float(np.nanmean(cos)),
                "shots": int(res["resources"]["shots"]),
            }
        )
    frame = pd.DataFrame(rows)
    out = args.pidn / "summary"
    out.mkdir(exist_ok=True)
    frame.to_csv(out / "shot_scaling_closed.csv", index=False)
    order = [p for p, _, _ in PLAN]
    mean = frame.groupby(["pipeline", "method"]).mean(numeric_only=True)
    methods = ["GD", *sorted(set(frame["method"]) - {"GD"})]
    lines = [
        f"# Closed loop on emulated Stage I data: {args.pidn.name}",
        "",
        "Final noise-free relative error after 60 iterations (mean of seeds 1-3); for PIDN in",
        "brackets the mean cosine of the surrogate gradient with the noise-free one in Stage III",
        "and the first iteration at which the error exceeds the Stage I error (mean). GD = the",
        "emulated gradient descent itself (60 measured gradients at the same shots).",
        "",
        "| pipeline | " + " | ".join(methods) + " | GD shots (M) | PIDN shots (M) |",
        "|---|" + "---|" * (len(methods) + 2),
    ]
    for p in order:
        if p not in mean.index.get_level_values(0):
            continue
        cells = []
        for m in methods:
            if (p, m) not in mean.index:
                cells.append("")
                continue
            r = mean.loc[(p, m)]
            cells.append(
                f"{r.final_error:.4f}"
                if m == "GD"
                else f"{r.final_error:.4f} ({r.stage3_cosine:+.2f}, t={r.first_worse:.0f})"
            )
        gd_shots = mean.loc[(p, "GD")].shots / 1e6 if (p, "GD") in mean.index else math.nan
        pidn = [mean.loc[(p, m)].shots for m in methods[1:] if (p, m) in mean.index]
        pidn_shots = max(pidn) / 1e6 if pidn else math.nan
        lines.append(f"| {p} | " + " | ".join(cells) + f" | {gd_shots:,.1f} | {pidn_shots:,.1f} |")
    text = "\n".join(lines) + "\n"
    (out / "shot_scaling_closed.md").write_text(text)
    sys.stdout.write(text)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--out", type=Path, default=None)
    g.add_argument("--only", nargs="*", default=None)
    o = sub.add_parser("openloop")
    o.add_argument("dir", type=Path)
    o.add_argument("config", type=Path)
    o.add_argument("--only", nargs="*", default=None)
    o.add_argument("--smooth", nargs="*", default=["pidn_paper", "quadratic"])
    o.add_argument("--real", action="store_true", help="Also test the real runs.")
    r = sub.add_parser("report")
    r.add_argument("dir", type=Path)
    c = sub.add_parser("closed")
    c.add_argument("dir", type=Path)
    c.add_argument("pidn", type=Path, help="kagome-vqe pidn study directory on DIR's runs.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    commands = {"generate": generate, "openloop": openloop, "report": report, "closed": closed}
    commands[args.command](args)


if __name__ == "__main__":
    main()
