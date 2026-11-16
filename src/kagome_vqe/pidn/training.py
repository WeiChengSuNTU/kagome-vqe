"""Training of the PIDN network: the composite loss, early stopping, ensembles, refreshes.

The loss of PIDN (arXiv:2605.02066, Eqs. 34-36) is ``L = L_data + beta L_phys`` with

    L_data = mean_t |C_hat(theta_t) - C_ZNE(theta_t)|^2 + |theta_hat_{t+1} - theta_{t+1}|^2
    L_phys = mean_t |grad_theta C_hat(theta_t) - g_ZNE(theta_t)|^2

Each term here is divided by the scale of its own target (``value_scale^2``, the mean squared
step, the mean squared gradient norm), so that ``beta`` and the term weights are unit-free and
mean the same thing on any problem; the paper's raw sum would let the energy term (``O(1)``
Hartree or here ``O(0.1)`` in units of ``J``) and the step term (``O(1e-4)``) differ by orders of
magnitude. Optimiser: AdamW with global-norm gradient clipping.

By default the network is trained on every Stage I step for a fixed number of epochs, as the
paper describes (it reports how well the training trajectory is fitted). Optionally
(``validation_fraction > 0``) generalisation is checked *forward in time*: the last part of
Stage I is held out, the epoch with the lowest held-out loss is kept (early stopping), and with
``refit`` the network is retrained on all steps for that many epochs. With shot-noise-dominated
ZNE gradients two held-out steps are too noisy to select on (it picks a handful of epochs, an
untrained network), which is why this is an ablation and not the default.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from kagome_vqe.pidn.autodiff import Tensor, grad
from kagome_vqe.pidn.model import (
    FloatArray,
    Normaliser,
    PidnModelConfig,
    PidnNetwork,
    TrainingBatch,
)

__all__ = ["PidnEnsemble", "PidnTrainingConfig", "pidn_losses"]


class PidnTrainingConfig(BaseModel):
    """Training hyper-parameters (not specified in the paper; see the module docstring).

    Attributes:
        epochs: Maximum full-batch epochs.
        learning_rate: AdamW step size.
        weight_decay: Decoupled weight decay.
        beta: Weight of the physics (gradient-consistency) loss; 0 is the data-only ablation.
        value_weight: Weight of the value term of ``L_data``.
        step_weight: Weight of the next-step term of ``L_data`` (0 without a step head).
        validation_fraction: Last part of the training phase held out for early stopping (0:
            train on everything for ``epochs``).
        patience: Epochs without held-out improvement before stopping.
        refit: Retrain on all steps for the selected number of epochs.
        ensemble: Independently initialised networks averaged at prediction.
        grad_clip: Global-norm clip of the weight gradient.
        refresh_epochs: Epochs of warm-started training after new anchor data.
        seed: Seeds the initialisations.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    epochs: int = Field(default=2000, ge=1)
    learning_rate: float = Field(default=3e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    beta: float = Field(default=1.0, ge=0)
    value_weight: float = Field(default=1.0, ge=0)
    step_weight: float = Field(default=1.0, ge=0)
    validation_fraction: float = Field(default=0.0, ge=0, lt=1)
    patience: int = Field(default=300, ge=1)
    refit: bool = True
    ensemble: int = Field(default=3, ge=1)
    grad_clip: float = Field(default=1.0, gt=0)
    refresh_epochs: int = Field(default=300, ge=1)
    seed: int = 0


def pidn_losses(
    network: PidnNetwork, batch: TrainingBatch, cfg: PidnTrainingConfig, *, create_graph: bool
) -> dict[str, Tensor]:
    """Normalised loss terms and their weighted total for ``batch``."""
    value, gradient, step = network.value_and_gradient(batch, create_graph=create_graph)
    norm = network.norm
    v_err = value - Tensor(batch.value[:, None])
    l_value = (v_err * v_err).mean() * (1.0 / norm.value_scale**2)
    g_err = gradient - Tensor(batch.gradient)
    g_scale = float(np.mean(np.sum(batch.gradient**2, axis=1))) or 1.0
    l_phys = (g_err * g_err).sum(axis=1).mean() * (1.0 / g_scale)
    total = l_value * cfg.value_weight + l_phys * cfg.beta
    terms = {"value": l_value, "physics": l_phys}
    if step is not None and cfg.step_weight > 0:
        s_err = step - Tensor(batch.step)
        s_scale = float(np.mean(np.sum(batch.step**2, axis=1))) or 1.0
        l_step = (s_err * s_err).sum(axis=1).mean() * (1.0 / s_scale)
        total = total + l_step * cfg.step_weight
        terms["step"] = l_step
    terms["total"] = total
    return terms


class _AdamW:
    def __init__(self, params: list[Tensor], lr: float, weight_decay: float, clip: float) -> None:
        self.params = params
        self.lr = lr
        self.wd = weight_decay
        self.clip = clip
        self.m = [np.zeros_like(p.value) for p in params]
        self.v = [np.zeros_like(p.value) for p in params]
        self.t = 0

    def step(self, grads: list[Tensor]) -> None:
        self.t += 1
        arrays = [g.value for g in grads]
        norm = math.sqrt(sum(float(np.sum(a * a)) for a in arrays))
        factor = min(1.0, self.clip / norm) if norm > 0 else 1.0
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, (p, raw) in enumerate(zip(self.params, arrays, strict=True)):
            g = raw * factor
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            m_hat = self.m[i] / (1 - b1**self.t)
            v_hat = self.v[i] / (1 - b2**self.t)
            p.value = p.value - self.lr * (m_hat / (np.sqrt(v_hat) + eps) + self.wd * p.value)


def _train(
    network: PidnNetwork,
    train: TrainingBatch,
    valid: TrainingBatch | None,
    cfg: PidnTrainingConfig,
    epochs: int,
    *,
    early_stopping: bool,
) -> tuple[list[dict[str, float]], int]:
    """Full-batch AdamW; returns the loss history and the selected epoch count."""
    params = network.parameters()
    opt = _AdamW(params, cfg.learning_rate, cfg.weight_decay, cfg.grad_clip)
    history: list[dict[str, float]] = []
    best = (math.inf, 0, network.get_state())
    for epoch in range(1, epochs + 1):
        terms = pidn_losses(network, train, cfg, create_graph=True)
        grads = grad(terms["total"], params)
        opt.step(grads)
        row = {"epoch": float(epoch), **{f"train_{k}": float(v.value) for k, v in terms.items()}}
        monitor = float(terms["total"].value)
        if valid is not None and len(valid):
            vterms = pidn_losses(network, valid, cfg, create_graph=False)
            row |= {f"valid_{k}": float(v.value) for k, v in vterms.items()}
            monitor = float(vterms["total"].value)
        history.append(row)
        if monitor < best[0]:
            best = (monitor, epoch, network.get_state())
        elif early_stopping and epoch - best[1] >= cfg.patience:
            break
    if early_stopping:
        network.set_state(best[2])
        return history, best[1]
    return history, epochs


class PidnEnsemble:
    """An ensemble of :class:`PidnNetwork` (``ensemble = 1`` is a single network).

    Args:
        model: Architecture.
        training: Training hyper-parameters.
    """

    def __init__(self, model: PidnModelConfig, training: PidnTrainingConfig) -> None:
        """Create the (unfitted) ensemble."""
        self.model = model
        self.training = training
        self.members: list[PidnNetwork] = []
        self.normaliser: Normaliser | None = None
        self.history: list[dict[str, float]] = []

    def fit(self, batch: TrainingBatch, **kwargs: Any) -> dict[str, Any]:
        """Normalise on ``batch``, then train every member (early stopping, optional refit)."""
        del kwargs
        cfg = self.training
        tick = time.perf_counter()
        self.normaliser = Normaliser.fit(batch, self.model.features)
        n = batch.theta.shape[1]
        holdout = round(cfg.validation_fraction * len(batch))
        holdout = min(holdout, len(batch) - 2) if holdout else 0
        train = batch.subset(slice(0, len(batch) - holdout)) if holdout else batch
        valid = batch.subset(slice(len(batch) - holdout, None)) if holdout else None
        self.members = []
        self.history = []
        selected = []
        for m in range(cfg.ensemble):
            seed = cfg.seed * 1000 + m
            net = PidnNetwork(self.model, n, self.normaliser, seed)
            hist, best_epoch = _train(net, train, valid, cfg, cfg.epochs, early_stopping=True)
            for row in hist:
                row["member"] = float(m)
                row["phase"] = 0.0
            self.history.extend(hist)
            if valid is not None and cfg.refit:
                net = PidnNetwork(self.model, n, self.normaliser, seed)
                hist, _ = _train(net, batch, None, cfg, max(best_epoch, 1), early_stopping=False)
                for row in hist:
                    row["member"] = float(m)
                    row["phase"] = 1.0
                self.history.extend(hist)
            selected.append(best_epoch)
            self.members.append(net)
        final = [pidn_losses(net, batch, cfg, create_graph=False) for net in self.members]
        return {
            "epochs_selected": selected,
            "train_seconds": time.perf_counter() - tick,
            "final_value_loss": float(np.mean([f["value"].value for f in final])),
            "final_physics_loss": float(np.mean([f["physics"].value for f in final])),
            "holdout_steps": holdout,
        }

    def refresh(self, batch: TrainingBatch) -> dict[str, Any]:
        """Warm-started training on an enlarged data set (same normalisation)."""
        tick = time.perf_counter()
        for m, net in enumerate(self.members):
            hist, _ = _train(
                net, batch, None, self.training, self.training.refresh_epochs, early_stopping=False
            )
            for row in hist:
                row["member"] = float(m)
                row["phase"] = 2.0
            self.history.extend(hist)
        return {"refresh_seconds": time.perf_counter() - tick}

    def predict(self, batch: TrainingBatch) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Ensemble means of value, gradient and step."""
        outputs = [net.predict(batch) for net in self.members]
        return tuple(np.mean([o[i] for o in outputs], axis=0) for i in range(3))  # type: ignore[return-value]

    def spread(self, batch: TrainingBatch) -> FloatArray:
        """Mean pairwise cosine disagreement of the members' gradients (0 = identical)."""
        grads = [net.predict(batch)[1] for net in self.members]
        if len(grads) < 2:
            return np.zeros(len(batch))
        mean = np.mean(grads, axis=0)
        cos = [
            np.sum(g * mean, axis=1) / (np.linalg.norm(g, axis=1) * np.linalg.norm(mean, axis=1))
            for g in grads
        ]
        return 1.0 - np.mean(cos, axis=0)
