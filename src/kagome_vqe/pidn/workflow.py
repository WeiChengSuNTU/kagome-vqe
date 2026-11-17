"""The three-stage PIDN workflow on saved ZNE gradient-descent runs, with full cost accounting.

Liu and Wang (arXiv:2605.02066, Sec. III C):

1. **Stage I, ZNE data collection.** ZNE-mitigated gradient descent for ``T_init`` iterations,
   recording ``(theta_{0:t}, C_noisy(theta_t), C_ZNE(theta_t), theta_{t+1}, g_ZNE(theta_t))``.
2. **Stage II, surrogate training** on those records (:mod:`kagome_vqe.pidn.training`).
3. **Stage III, surrogate-driven optimisation**: ``theta_{t+1} = theta_t - eta grad_theta
   F(theta_{0:t}, C_noisy(theta_t))`` (Eq. 38). The only quantum cost per step is the noisy cost
   ``C_noisy(theta_t)`` at noise scale 1 (one energy: the measurement bases, no ZNE).

Stage I is not re-run here: it is, by construction, the first ``T_init`` iterations of a ZNE
gradient-descent run with the same seed, so this module reads them from a saved run's
``trajectory.npz`` (``configs/studies/graph_hva_zne_gd_representative.toml``) and charges their
circuit executions and shots to the PIDN run exactly as recorded. Stage III then continues on
the same backend (same noise model, shots, ZNE settings, learning rate) with fresh randomness.

Additions beyond the paper, all off unless configured:

* ``max_step_factor``: cap every surrogate step at that multiple of the median Stage I step
  (a trust region; extrapolating surrogates otherwise can take arbitrarily large steps);
* ``anchor_every``: every ``K`` surrogate steps, measure a real ZNE gradient (full cost), use
  it for that step, add it to the training data and refresh the surrogate (the "adaptive
  online retraining" the paper lists as future work);
* non-neural surrogates (``quadratic``, ``last_gradient``) through the same loop, which need no
  quantum evaluation at all in Stage III.

Diagnostics that a real experiment could not afford -- the exact noise-free energy and gradient
at every iterate -- are computed classically, timed separately and never charged as cost.
"""

from __future__ import annotations

import contextlib
import glob
import json
import logging
import math
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
import tomli_w
from pydantic import BaseModel, ConfigDict, Field, model_validator

from kagome_vqe.config import load_toml
from kagome_vqe.pidn.model import (
    LastGradientSurrogate,
    PidnModelConfig,
    QuadraticSurrogate,
    Surrogate,
    build_batch,
)
from kagome_vqe.pidn.training import PidnEnsemble, PidnTrainingConfig

__all__ = [
    "PidnStage3Config",
    "PidnStudyConfig",
    "PidnVariant",
    "load_pidn_config",
    "run_pidn_study",
    "run_pidn_variant",
    "summarise_pidn_study",
]

log = logging.getLogger(__name__)

_LABEL = r"^[A-Za-z0-9_.]+$"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PidnStage3Config(_Frozen):
    """How the surrogate drives the optimisation.

    Attributes:
        update: ``"value_gradient"`` (Eq. 38: step along ``-grad C_hat``) or ``"step_head"``
            (use the predicted ``theta_hat_{t+1}`` directly).
        max_step_factor: Cap each step at this multiple of the median Stage I step (``None``:
            no cap, as in the paper).
        anchor_every: Measure a real ZNE gradient every this many surrogate steps (``None``:
            never, as in the paper).
        measure_noisy: Measure ``C_noisy(theta_t)`` every step (needed by a PIDN with a noisy
            input; the baselines do not use it, so it can be switched off to make their
            Stage III free).
    """

    update: Literal["value_gradient", "step_head"] = "value_gradient"
    max_step_factor: Annotated[float, Field(gt=0)] | None = None
    anchor_every: Annotated[int, Field(ge=1)] | None = None
    measure_noisy: bool = True


class PidnVariant(_Frozen):
    """One surrogate configuration of the comparison.

    Attributes:
        label: Name in tables and run directories.
        model: Architecture (or baseline kind).
        training: Training hyper-parameters (ignored by the baselines).
        stage3: Surrogate-phase settings.
        t_init: Stage I length for this variant (``None``: the study's ``t_init``).
    """

    label: Annotated[str, Field(pattern=_LABEL)]
    model: PidnModelConfig = PidnModelConfig()
    training: PidnTrainingConfig = PidnTrainingConfig()
    stage3: PidnStage3Config = PidnStage3Config()
    t_init: Annotated[int, Field(ge=2)] | None = None

    @model_validator(mode="after")
    def _consistent(self) -> PidnVariant:
        if self.stage3.update == "step_head" and not (
            self.model.kind == "pidn" and self.model.step_head
        ):
            msg = "stage3.update = 'step_head' needs a pidn model with step_head = true"
            raise ValueError(msg)
        if (
            self.model.kind == "pidn"
            and self.model.noisy_hidden > 0
            and not (self.stage3.measure_noisy)
        ):
            msg = "a pidn model with a noisy input needs stage3.measure_noisy = true"
            raise ValueError(msg)
        return self


class PidnStudyConfig(_Frozen):
    """A PIDN comparison (TOML in ``configs/pidn/``).

    Attributes:
        name: Output directory suffix.
        description: Free text for the report.
        sources: ZNE gradient-descent run directories or glob patterns (one PIDN run per
            source and variant); each needs ``trajectory.npz`` with at least ``t_init`` rows.
        root: Base directory for relative patterns.
        t_init: Stage I iterations taken from the source.
        total_iterations: Total iterations (Stage I + Stage III).
        learning_rate: Stage III step size; ``None`` uses the source run's.
        track_ideal: Noise-free energy and gradient at every iterate (diagnostics).
        final_zne_shots: Shots per basis of the final ZNE re-measurement (``None``: the source's
            ``backend.final_shots``).
        seed_offset: Stage III randomness is seeded by the source seed plus this offset (the
            same for every variant, so variants see common random numbers).
        variants: Surrogates to compare.
        baseline_studies: Gradient-descent study directories (globs) shown as baselines in the
            report; default: the studies the sources belong to.
    """

    name: str = "graph_hva_pidn"
    description: str = ""
    sources: list[str]
    root: Path | None = None
    output_dir: Path = Path("results")
    t_init: Annotated[int, Field(ge=2)] = 10
    total_iterations: Annotated[int, Field(ge=3)] = 60
    learning_rate: Annotated[float, Field(gt=0)] | None = None
    track_ideal: bool = True
    final_zne_shots: Annotated[int, Field(ge=1)] | None = None
    seed_offset: int = 7919
    variants: Annotated[list[PidnVariant], Field(min_length=1)]
    baseline_studies: list[str] | None = None

    @model_validator(mode="after")
    def _checks(self) -> PidnStudyConfig:
        longest = max([self.t_init, *(v.t_init or 0 for v in self.variants)])
        if self.total_iterations <= longest:
            msg = "total_iterations must exceed every t_init"
            raise ValueError(msg)
        labels = [v.label for v in self.variants]
        if len(set(labels)) != len(labels):
            msg = f"duplicate variant labels {labels}"
            raise ValueError(msg)
        return self


def load_pidn_config(path: Path) -> PidnStudyConfig:
    """Read a PIDN study TOML."""
    return PidnStudyConfig.model_validate(load_toml(Path(path)))


def resolve_pidn_sources(cfg: PidnStudyConfig) -> list[Path]:
    """Source run directories with a ``trajectory.npz``."""
    base = Path.cwd() if cfg.root is None else Path(cfg.root)
    found: list[Path] = []
    for pattern in cfg.sources:
        full = pattern if Path(pattern).is_absolute() else str(base / pattern)
        for match in sorted(glob.glob(full)):
            path = Path(match)
            if (path / "trajectory.npz").exists() and path not in found:
                found.append(path)
    if not found:
        msg = f"no runs with trajectory.npz match {cfg.sources} (relative to {base})"
        raise FileNotFoundError(msg)
    return found


def _surrogate(variant: PidnVariant) -> Surrogate:
    kind = variant.model.kind
    if kind == "quadratic":
        return QuadraticSurrogate(variant.model.quadratic_ridge)
    if kind == "last_gradient":
        return LastGradientSurrogate()
    return PidnEnsemble(variant.model, variant.training)


class _Quantum:
    """The source run's backend: noisy energies, ZNE gradients, final ZNE re-measurement."""

    def __init__(self, source_cfg: Any, seed: int, *, noisy_mitigation: Any = None) -> None:
        """Open the estimators.

        ``noisy_mitigation`` is the mitigation of the estimator behind ``C_noisy`` (PIDN's input):
        it must be the one that produced the Stage I noisy column (see
        :func:`noisy_input_mitigation`); ``None`` means unmitigated.
        """
        from kagome_vqe.estimators import make_estimator
        from kagome_vqe.vqe import (
            EnergyObjective,
            _ideal_energy_fn,
            _ideal_gradient_fn,
            prepare,
        )

        self.cfg = source_cfg
        self.prep = prepare(source_cfg)
        self.map = self.prep.parameter_map
        self.groups = self.prep.problem.measurement_groups
        zne = source_cfg.mitigation.zne
        self.scales = len(zne.scales) if zne is not None else 1
        exact = source_cfg.backend.is_exact
        self.shots = 0 if exact else int(source_cfg.backend.shots or 0)
        if exact:
            self.groups = 0  # exact energies: no circuits are counted
        self._noisy = make_estimator(
            source_cfg.backend,
            self.prep.problem,
            None,
            seed=seed,
            mitigation=noisy_mitigation,
            twin=self.prep.circuit_twin,
        )
        self._zne = make_estimator(
            source_cfg.backend,
            self.prep.problem,
            None,
            seed=seed + 1,
            mitigation=source_cfg.mitigation,
            twin=self.prep.circuit_twin,
        )
        # Opened for the whole run (a twin is measured once, on entry); see close().
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(self._noisy)
        self._stack.enter_context(self._zne)
        self._objective = EnergyObjective(self._zne, parameter_map=self.map)
        tracked = replace(
            self.prep, config=source_cfg.model_copy(update={"track_ideal_gradient": True})
        )
        self.ideal_gradient = _ideal_gradient_fn(tracked)
        self.ideal_energy = _ideal_energy_fn(self.prep)

    def close(self) -> None:
        """Release the estimators."""
        self._stack.close()

    def noisy_energy(self, theta: np.ndarray) -> tuple[float, float]:
        batch = self._noisy.evaluate(self.map.expand(theta)[None, :])
        return float(batch.energies[0]), float(batch.stds[0])

    def zne_gradient(self, theta: np.ndarray) -> Any:
        from kagome_vqe.optimizers import ParameterShift

        return ParameterShift(self.map, include_value=True).estimate(self._objective, theta)

    def final(self, theta: np.ndarray, shots: int) -> dict[str, Any]:
        batch = self._zne.evaluate(
            self.map.expand(theta)[None, :], shots=shots if self.shots else None
        )
        scales = batch.scales or {"energies": batch.energies[None, :]}
        raw = np.asarray(scales["energies"])[:, 0]
        return {
            "final_zne_energy": float(batch.energies[0]),
            "final_zne_std": float(batch.stds[0]),
            "final_unmitigated_energy": float(raw[0]),
            "final_shots": shots,
        }


def noisy_input_mitigation(run: Any) -> Any:
    """Mitigation of the estimator that produced a source's noisy column ``scale_values[:, 0]``.

    Stage III must measure ``C_noisy`` the same way, or the surrogate's noisy input leaves its
    training distribution (an unmitigated energy sits ~10 % above a readout+symmetry one).

    * Relabelled runs (:mod:`kagome_vqe.pidn.relabel`) measure it unmitigated.
    * ZNE runs record the scale-1 energies of the ZNE estimator's inner stack: the counts-level
      mitigation (readout, symmetry), without extrapolation or rescaling.
    * Other gradient-descent runs record the value itself: the full mitigation stack.
    """
    from kagome_vqe.config import MitigationConfig

    if "relabelled_from" in run.result:
        return MitigationConfig()
    mitigation = run.config.mitigation
    if mitigation.zne is not None:
        return mitigation.model_copy(update={"zne": None, "rescale": None})
    return mitigation


def gradient_variances(traj: dict[str, np.ndarray], mapping: Any) -> np.ndarray:
    """Shot-noise variance of every recorded parameter-shift gradient component, ``(T, n)``.

    ``g_i = sum_lambda w_lambda sum_{k in i} c_k (E_lambda(phi + s_k) - E_lambda(phi - s_k))``,
    so ``var g_i = 2 (sum_{k in i} c_k^2) sum_lambda w_lambda^2 sigma_lambda^2`` with
    ``sigma_lambda`` the standard error of one energy at scale ``lambda``. The recorded standard
    error at the current point stands in for that of the shifted energies (they lie within
    ``pi/4`` of it and use the same shots). Without ZNE there is one scale with ``w = 1``.
    """
    c2 = np.asarray(mapping.occurrence_coefficients, dtype=float) ** 2
    per_param = np.bincount(
        mapping.occurrence_to_param, weights=c2, minlength=mapping.num_parameters
    )
    weights = np.asarray(traj["zne_weights"], dtype=float)
    stds = np.asarray(traj["scale_stds"], dtype=float)
    energy_var = np.sum(weights**2 * stds**2, axis=1)
    return 2.0 * energy_var[:, None] * per_param[None, :]


def _source_arrays(source: Path, t_init: int) -> tuple[Any, dict[str, np.ndarray]]:
    from kagome_vqe.results import load_run

    run = load_run(source)
    traj = run.trajectory
    if traj is None or len(traj["value"]) < t_init:
        msg = f"{source}: needs a trajectory with at least {t_init} iterations"
        raise ValueError(msg)
    cfg = run.config
    if cfg.optimizer.type != "gd":
        msg = f"{source}: Stage I must be gradient descent with a recorded trajectory"
        raise ValueError(msg)
    if cfg.mitigation.zne is None:
        log.info(
            "%s: Stage I without ZNE (%s); the surrogate learns that run's gradients",
            run.name,
            cfg.backend.label(),
        )
    return run, traj


class _PidnRun:
    """State of one PIDN run: Stage I data, the surrogate, the growing trajectory."""

    def __init__(self, cfg: PidnStudyConfig, variant: PidnVariant, source: Path) -> None:
        self.t0 = variant.t_init or cfg.t_init
        run, traj = _source_arrays(source, self.t0)
        self.cfg, self.variant, self.source, self.run, self.traj = cfg, variant, source, run, traj
        self.lr = cfg.learning_rate or float(traj["learning_rate"])
        self.q = _Quantum(
            run.config,
            run.config.seed + cfg.seed_offset,
            noisy_mitigation=noisy_input_mitigation(run),
        )
        self.n = traj["theta"].shape[1]
        self.window = variant.model.history
        t0 = self.t0
        # Stage I, as recorded by the source run (its cost is charged below).
        self.thetas = [traj["theta"][t].copy() for t in range(t0)] + [
            traj["theta_next"][t0 - 1].copy()
        ]
        self.noisy = [float(v) for v in traj["scale_values"][:t0, 0]]
        self.values = traj["value"][:t0].copy()
        self.gradients = traj["grad"][:t0].copy()
        self.steps = traj["theta_next"][:t0] - traj["theta"][:t0]
        self.train_rows = list(range(t0))
        self.grad_var: np.ndarray | None = None
        if variant.training.noise_aware:
            self.grad_var = gradient_variances(traj, self.q.map)[:t0]
            if not np.all(np.isfinite(self.grad_var)):
                msg = f"{source}: Stage I has no standard errors for the noise-aware loss"
                raise ValueError(msg)
        self.executions = int(traj["circuit_executions"][t0 - 1])
        self.shots = int(traj["shots"][t0 - 1])
        self.stage1 = {"executions": self.executions, "shots": self.shots}
        factor = variant.stage3.max_step_factor
        median_step = float(np.median(np.linalg.norm(self.steps, axis=1)))
        self.max_step = None if factor is None else factor * median_step
        self.rows = [self._stage1_row(t) for t in range(t0)]
        self.counters = {"stage3": 0, "anchor": 0, "anchors": 0}
        self.timers = {"train": 0.0, "stage3": 0.0, "diagnostic": 0.0}
        self.surrogate = _surrogate(variant)
        self.fit_info: dict[str, Any] = {}

    def _stage1_row(self, t: int) -> dict[str, Any]:
        traj = self.traj
        return {
            "iteration": t,
            "phase": 0,
            "theta": traj["theta"][t],
            "theta_next": traj["theta_next"][t],
            "grad": traj["grad"][t],
            "surrogate_grad": np.full(self.n, np.nan),
            "value": float(traj["value"][t]),
            "noisy": float(traj["scale_values"][t, 0]),
            "ideal_energy": float(traj["ideal_energy"][t]),
            "ideal_grad": traj["ideal_grad"][t],
            "circuit_executions": int(traj["circuit_executions"][t]),
            "shots": int(traj["shots"][t]),
            "anchor_cosine": math.nan,
            "ensemble_spread": math.nan,
        }

    def _training_batch(self, thetas: np.ndarray, noisy: np.ndarray) -> Any:
        length = len(noisy)
        return build_batch(
            thetas,
            noisy,
            window=self.window,
            values=_scatter_rows(self.values, self.train_rows, length),
            gradients=_scatter_rows(self.gradients, self.train_rows, length),
            steps=_scatter_rows(self.steps, self.train_rows, length),
            rows=self.train_rows,
            gradient_variances=None
            if self.grad_var is None
            else _scatter_rows(self.grad_var, self.train_rows, length),
        )

    def train(self) -> None:
        """Stage II."""
        tick = time.perf_counter()
        batch = self._training_batch(np.vstack(self.thetas[: self.t0]), np.asarray(self.noisy))
        self.fit_info = self.surrogate.fit(batch)
        self.timers["train"] += time.perf_counter() - tick

    def _anchor(
        self, theta: np.ndarray, s_grad: np.ndarray
    ) -> tuple[np.ndarray, float, float, float]:
        """A real ZNE gradient at ``theta`` (full cost): gradient, value, cosine, noisy value."""
        q = self.q
        est = q.zne_gradient(theta)
        cost = est.nfev * q.groups * q.scales
        self.executions += cost
        self.counters["anchor"] += cost
        self.counters["anchors"] += 1
        self.shots += cost * q.shots
        cos = float(np.dot(s_grad, est.grad) / (np.linalg.norm(s_grad) * np.linalg.norm(est.grad)))
        noisy = float(est.details["scale_values"][0]) if est.details else math.nan
        value = math.nan if est.value is None else float(est.value)
        return np.asarray(est.grad, dtype=float), value, cos, noisy

    def step(self, t: int) -> bool:
        """One Stage III iteration at ``theta_t``; returns False if the parameters diverged."""
        tick = time.perf_counter()
        q, variant = self.q, self.variant
        theta = self.thetas[-1].copy()
        c_t = math.nan
        if variant.stage3.measure_noisy:
            c_t, _ = q.noisy_energy(theta)
            self.executions += q.groups
            self.counters["stage3"] += q.groups
            self.shots += q.groups * q.shots
        query = build_batch(
            np.vstack(self.thetas), np.asarray([*self.noisy, c_t]), window=self.window, rows=[t]
        )
        value, s_grad, s_step = (a[0] for a in self.surrogate.predict(query))
        spread = (
            float(self.surrogate.spread(query)[0])
            if isinstance(self.surrogate, PidnEnsemble)
            else math.nan
        )
        used, phase, anchor_cos = np.array(s_grad, dtype=float), 1, math.nan
        every = variant.stage3.anchor_every
        if every is not None and (t - self.t0 + 1) % every == 0:
            used, value, anchor_cos, c_anchor = self._anchor(theta, s_grad)
            phase = 2
            c_t = c_anchor if math.isnan(c_t) else c_t
        if variant.stage3.update == "step_head" and phase == 1:
            delta = np.asarray(s_step, dtype=float)
        else:
            delta = -self.lr * used
        norm = float(np.linalg.norm(delta))
        if self.max_step is not None and phase == 1 and norm > self.max_step:
            delta *= self.max_step / norm
        new_theta = theta + delta
        if phase == 2:
            self.train_rows.append(t)
            self.values = np.append(self.values, value)
            self.gradients = np.vstack([self.gradients, used[None, :]])
            if self.grad_var is not None:  # same estimator and shots as Stage I
                self.grad_var = np.vstack([self.grad_var, self.grad_var[: self.t0].mean(axis=0)])
            self.steps = np.vstack([self.steps, delta[None, :]])
            batch = self._training_batch(np.vstack(self.thetas), np.asarray([*self.noisy, c_t]))
            if isinstance(self.surrogate, PidnEnsemble):
                self.surrogate.refresh(batch)
            else:
                self.surrogate.fit(batch)
        self.timers["stage3"] += time.perf_counter() - tick
        ideal_e, ideal_g = self._diagnostics(theta)
        self.rows.append(
            {
                "iteration": t,
                "phase": phase,
                "theta": theta,
                "theta_next": new_theta.copy(),
                "grad": -delta / self.lr,
                "surrogate_grad": np.asarray(s_grad, dtype=float),
                "value": float(value),
                "noisy": c_t,
                "ideal_energy": ideal_e,
                "ideal_grad": ideal_g,
                "circuit_executions": self.executions,
                "shots": self.shots,
                "anchor_cosine": anchor_cos,
                "ensemble_spread": spread,
            }
        )
        self.noisy.append(c_t)
        self.thetas.append(new_theta)
        return bool(np.all(np.isfinite(new_theta)))

    def _diagnostics(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """Noise-free energy and gradient (classical bookkeeping, not charged)."""
        tick = time.perf_counter()
        q = self.q
        track = self.cfg.track_ideal
        energy = q.ideal_energy(theta) if track and q.ideal_energy else math.nan
        gradient = (
            q.ideal_gradient(theta) if track and q.ideal_gradient else np.full(self.n, np.nan)
        )
        self.timers["diagnostic"] += time.perf_counter() - tick
        return float(energy), np.asarray(gradient, dtype=float)

    def result(self) -> dict[str, Any]:
        """Final re-measurement and the run's summary."""
        q, cfg = self.q, self.cfg
        theta = self.thetas[-1]
        finite = bool(np.all(np.isfinite(theta)))
        source_cfg = self.run.config
        final_shots = cfg.final_zne_shots or source_cfg.backend.final_shots or q.shots
        final = q.final(theta, final_shots) if finite else {}
        ideal = q.ideal_energy(theta) if q.ideal_energy and finite else None
        exact = float(self.run.result["exact_energy"])
        return {
            "variant": self.variant.label,
            "source_pipeline": self.run.result.get("pipeline"),
            "source": str(self.source),
            "source_name": self.run.name,
            "seed": source_cfg.seed,
            "t_init": self.t0,
            "total_iterations": len(self.rows),
            "learning_rate": self.lr,
            "exact_energy": exact,
            "final_ideal_energy": ideal,
            "final_ideal_relative_error": None
            if ideal is None
            else abs(ideal - exact) / abs(exact),
            **final,
            "resources": {
                "stage1_circuit_executions": self.stage1["executions"],
                "stage1_shots": self.stage1["shots"],
                "stage3_circuit_executions": self.counters["stage3"],
                "anchor_circuit_executions": self.counters["anchor"],
                "anchors": self.counters["anchors"],
                "circuit_executions": self.executions,
                "shots": self.shots,
                "final_circuit_executions": q.groups * q.scales,
                "train_seconds": self.timers["train"],
                "stage3_seconds": self.timers["stage3"],
                "diagnostic_seconds": self.timers["diagnostic"],
            },
            "training": self.fit_info,
            "variant_config": self.variant.model_dump(mode="json"),
        }


def run_pidn_variant(
    cfg: PidnStudyConfig, variant: PidnVariant, source: Path, out_dir: Path
) -> dict[str, Any]:
    """One PIDN (or baseline) run continuing the ZNE run ``source``; writes ``out_dir``."""
    state = _PidnRun(cfg, variant, source)
    try:
        state.train()
        for t in range(state.t0, cfg.total_iterations):
            if not state.step(t):
                log.warning("%s: non-finite parameters at iteration %d; stopping", variant.label, t)
                break
        result = state.result()
    finally:
        state.q.close()
    _save_run(out_dir, state.rows, result, state.surrogate, state.lr)
    return result


def _scatter_rows(data: np.ndarray, rows: list[int], length: int) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    out = np.full((length, *data.shape[1:]), np.nan)
    out[rows] = data
    return out


def _save_run(
    out_dir: Path, rows: list[dict[str, Any]], result: dict[str, Any], surrogate: Any, lr: float
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stack = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    trajectory = {
        "theta": stack["theta"],
        "theta_next": stack["theta_next"],
        "grad": stack["grad"],
        "surrogate_grad": stack["surrogate_grad"],
        "value": stack["value"],
        "value_std": np.full(len(rows), np.nan),
        "scales": np.ones(1),
        "scale_values": stack["noisy"][:, None],
        "scale_stds": np.full((len(rows), 1), np.nan),
        "scale_grads": np.full((len(rows), 1, stack["grad"].shape[1]), np.nan),
        "zne_weights": np.ones((len(rows), 1)),
        "learning_rate": np.array(lr),
        "evaluations": np.full(len(rows), -1),
        "circuit_executions": stack["circuit_executions"],
        "shots": stack["shots"],
        "ideal_energy": stack["ideal_energy"],
        "ideal_grad": stack["ideal_grad"],
        "phase": stack["phase"],
        "anchor_cosine": stack["anchor_cosine"],
        "ensemble_spread": stack["ensemble_spread"],
    }
    np.savez(out_dir / "trajectory.npz", **trajectory)  # type: ignore[arg-type]
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    history = getattr(surrogate, "history", None)
    if history:
        pd.DataFrame(history).to_csv(out_dir / "training.csv", index=False)


# -- study --------------------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def run_pidn_study(
    cfg: PidnStudyConfig,
    *,
    study_dir: Path | None = None,
    only: list[str] | None = None,
    summarise: bool = True,
) -> Path:
    """Every (variant, source) pair not yet completed in ``study_dir`` (new directory if None)."""
    from kagome_vqe.results import environment_info

    sources = resolve_pidn_sources(cfg)
    if study_dir is None:
        study_dir = Path(cfg.output_dir) / f"{_now()}-{cfg.name}"
        study_dir.mkdir(parents=True)
        with (study_dir / "pidn.toml").open("wb") as fh:
            tomli_w.dump(cfg.model_dump(mode="json", exclude_none=True), fh)
        (study_dir / "environment.json").write_text(json.dumps(environment_info(), indent=2) + "\n")
        (study_dir / "sources.json").write_text(
            json.dumps([str(s) for s in sources], indent=2) + "\n"
        )
    manifest = study_dir / "manifest.jsonl"
    done = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            row = json.loads(line)
            if row.get("status") == "completed":
                done.add(row["name"])
    for variant in cfg.variants:
        for source in sources:
            from kagome_vqe.results import load_run

            loaded = load_run(source)
            seed = loaded.config.seed
            pipeline = loaded.result.get("pipeline")  # relabelled Stage I (pidn.relabel)
            name = (
                f"{variant.label}-s{seed}"
                if pipeline is None
                else (f"{variant.label}-{pipeline}-s{seed}")
            )
            if name in done or (only and not any(o in name for o in only)):
                continue
            out = study_dir / "runs" / name
            log.info("PIDN %s from %s", name, source.name)
            row = {"name": name, "variant": variant.label, "seed": seed, "path": f"runs/{name}"}
            try:
                result = run_pidn_variant(cfg, variant, source, out)
            except Exception as exc:  # recorded, then re-raised
                row |= {"status": "failed", "error": repr(exc)}
                with manifest.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                raise
            row |= {
                "status": "completed",
                "final_ideal_relative_error": result["final_ideal_relative_error"],
            }
            with manifest.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
    if summarise:
        summarise_pidn_study(study_dir)
    return study_dir


def summarise_pidn_study(study_dir: Path) -> Path:
    """Write ``summary/`` (tables, report, figures) of a PIDN study."""
    from kagome_vqe.pidn.report import write_pidn_summary

    return write_pidn_summary(Path(study_dir))
