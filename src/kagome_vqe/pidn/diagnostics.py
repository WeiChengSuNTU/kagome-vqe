"""Open-loop extrapolation test of trajectory surrogates (no quantum evaluations).

Stage III of PIDN is closed-loop: the surrogate's own steps decide where it is queried next, so
a poor prediction also moves the trajectory somewhere the surrogate has never seen. This test
separates the two effects. A surrogate is trained on the first ``t_init`` iterations of a
recorded trajectory and then queried *along the recorded continuation* (teacher forcing):
the predicted gradient at every later recorded point is compared with the exact noise-free
gradient there (and with the recorded gradient). It answers "how far ahead does the training
data determine the gradient field?" independently of the feedback of Stage III.

Run it on a noise-free trajectory to see the best case (exact training targets), and on a ZNE
trajectory to see the effect of noisy targets.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kagome_vqe.pidn.model import build_batch
from kagome_vqe.pidn.workflow import PidnVariant, _surrogate
from kagome_vqe.trajectory_analysis import cosine

__all__ = ["extrapolation_test"]


def extrapolation_test(
    trajectory: dict[str, np.ndarray],
    variants: list[PidnVariant],
    *,
    t_init: int,
    horizon: int | None = None,
) -> pd.DataFrame:
    """Train each variant on ``t_init`` recorded steps; score it on the recorded continuation.

    Args:
        trajectory: A ``trajectory.npz`` dictionary (gradient descent with
            ``track_ideal_gradient``).
        variants: Surrogates to compare (their Stage III settings are ignored).
        t_init: Training steps.
        horizon: Last step scored (default: the end of the trajectory).

    Returns:
        One row per (variant, step ``t >= t_init``): cosine and norm ratio of the predicted
        gradient against the exact noise-free gradient and against the recorded gradient.
    """
    theta = np.vstack([trajectory["theta"], trajectory["theta_next"][-1:]])
    noisy = trajectory["scale_values"][:, 0]
    values, grads = trajectory["value"], trajectory["grad"]
    steps = trajectory["theta_next"] - trajectory["theta"]
    ideal = trajectory["ideal_grad"]
    end = len(values) if horizon is None else min(horizon, len(values))
    rows: list[dict[str, Any]] = []
    for variant in variants:
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
        query = build_batch(theta[:end], noisy[:end], window=window, rows=list(range(t_init, end)))
        _, predicted, _ = surrogate.predict(query)
        cos_ideal = cosine(predicted, ideal[t_init:end])
        cos_recorded = cosine(predicted, grads[t_init:end])
        ratio = np.linalg.norm(predicted, axis=1) / np.linalg.norm(ideal[t_init:end], axis=1)
        rows.extend(
            {
                "variant": variant.label,
                "step": t,
                "ahead": t - t_init + 1,
                "cosine_to_ideal": float(cos_ideal[i]),
                "cosine_to_recorded": float(cos_recorded[i]),
                "norm_ratio_to_ideal": float(ratio[i]),
                "train_seconds": seconds,
            }
            for i, t in enumerate(range(t_init, end))
        )
    return pd.DataFrame(rows)


def load_trajectory(run_dir: Path) -> dict[str, np.ndarray]:
    """``trajectory.npz`` of a run directory as a dictionary."""
    with np.load(Path(run_dir) / "trajectory.npz") as data:
        return {k: data[k] for k in data.files}
