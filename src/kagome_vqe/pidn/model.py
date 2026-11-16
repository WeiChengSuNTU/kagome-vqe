"""PIDN surrogate models: the physics-informed denoising network and non-neural baselines.

The physics-informed denoising network (PIDN) of Liu and Wang, arXiv:2605.02066, is a surrogate
``F_phi(theta_{0:t}, C_noisy(theta_t)) -> (C_hat(theta_t), theta_hat_{t+1})`` of
zero-noise-extrapolated (ZNE) gradient descent. The paper specifies its structure only
qualitatively (Sec. III B): "a feedforward branch that takes the noisy expectation value as
input" and "a recurrent branch implemented using a small GRU that encodes the historical
parameter trajectory", "fused and mapped to two output heads". Widths, depths, activations,
input encodings, normalisation and training hyper-parameters are not given. Every one of those
choices is a field of :class:`PidnModelConfig`, and the defaults below are reasoned rather than
copied (see ``docs/experiments/graph_hva_pidn.md`` for the argument and the ablations):

* **Smooth activations** (``tanh`` default). The physics loss ``||grad_theta C_hat - g_ZNE||^2``
  is trained through the input gradient; with ReLU that gradient is piecewise constant in
  ``theta`` and its weight derivative misses all curvature, so ReLU is an ablation only.
* **Input encodings.** ``raw``: ``(theta - theta_0) / s`` with one global scale (keeps the
  geometry; no per-parameter z-scoring, which explodes on parameters that barely moved during
  the training phase). ``fourier``: ``(cos 2 theta, sin 2 theta)``. Every exchange-gate angle
  enters the energy as a single frequency 2 (generator gap 2, see
  :mod:`kagome_vqe.ansatz.unshare`), so these features are exact for one-parameter cuts and
  bounded when the trajectory leaves the training region.
* **Recurrent input** per step: features of ``theta_k``, the step ``theta_k - theta_{k-1}``
  (scaled), and the noisy cost ``C_noisy(theta_k)``, over a window of ``history`` steps.
* **Direct path.** ``theta_t`` also enters the fusion trunk directly (``direct_theta``), so the
  value head is an explicit function of the current point and its input gradient does not
  have to pass through the recurrent cell alone.
* **Next-step head** predicts the *increment* ``theta_{t+1} - theta_t`` (scaled); the paper's
  loss ``||theta_hat_{t+1} - theta_{t+1}||^2`` is the same number, but the increment is ~1e-2
  of ``theta`` so predicting ``theta_{t+1}`` itself would be badly conditioned.
* **Output scale.** ``C_hat = mu + sigma v``. ``sigma`` is at least the gradient magnitude
  times the input-feature scale, so that unit weights can express the measured gradients (a
  pure ``std(C)`` scale can be ~0.1 on a flat stretch while ``|g|`` is ~3, which would force
  large weights).

Baselines without a neural network, fitted to the same data:

* :class:`QuadraticSurrogate`: ``C(theta) = c + g.d + d.H.d / 2`` around the last training
  point, ridge-fitted to the ZNE values and gradients (a learned quasi-Newton model);
* :class:`LastGradientSurrogate`: keeps stepping along the last ZNE gradient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from kagome_vqe.pidn.autodiff import (
    Tensor,
    concat,
    cos,
    grad,
    no_grad,
    relu,
    sigmoid,
    silu,
    sin,
    tanh,
)

__all__ = [
    "LastGradientSurrogate",
    "PidnModelConfig",
    "PidnNetwork",
    "QuadraticSurrogate",
    "Surrogate",
    "TrainingBatch",
    "build_batch",
]

FloatArray = NDArray[np.float64]

_ACTIVATIONS = {"tanh": tanh, "silu": silu, "relu": relu}


class PidnModelConfig(BaseModel):
    """Architecture of the surrogate (see the module docstring for the reasoning).

    Attributes:
        kind: ``"pidn"`` (the network), ``"quadratic"`` or ``"last_gradient"`` (baselines).
        features: ``"raw"`` or ``"fourier"`` encoding of the parameters.
        history: Steps of the trajectory the recurrent branch sees (window, padded at the
            start by repeating the first point).
        gru_hidden: GRU state size (0 disables the recurrent branch).
        noisy_hidden: Width of the feed-forward branch on the noisy cost (0 disables the
            noisy input entirely, also in the recurrent inputs).
        trunk: Widths of the fusion MLP.
        activation: ``tanh``, ``silu`` or ``relu`` (ReLU only as an ablation).
        direct_theta: Feed the features of ``theta_t`` to the trunk directly.
        step_head: Train the next-step head (the paper's two-head design).
        init_scale: Multiplies the Glorot initialisation of every weight matrix.
        quadratic_ridge: ``quadratic``: ridge strength on the Hessian (relative).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["pidn", "quadratic", "last_gradient"] = "pidn"
    features: Literal["raw", "fourier"] = "raw"
    history: int = Field(default=8, ge=1)
    gru_hidden: int = Field(default=32, ge=0)
    noisy_hidden: int = Field(default=16, ge=0)
    trunk: list[int] = Field(default_factory=lambda: [64, 64])
    activation: Literal["tanh", "silu", "relu"] = "tanh"
    direct_theta: bool = True
    step_head: bool = True
    init_scale: float = Field(default=1.0, gt=0)
    quadratic_ridge: float = Field(default=1e-2, ge=0)


@dataclass(slots=True)
class TrainingBatch:
    """Inputs and targets of a set of trajectory steps (one row per step ``t``).

    Attributes:
        theta: ``theta_t``, shape ``(B, n)``.
        history: ``theta_{t-W..t}``, shape ``(B, W + 1, n)`` (last row equals ``theta``; the
            extra first row defines the step into the window's first point).
        history_noisy: ``C_noisy`` along the same rows, shape ``(B, W + 1)``.
        noisy: ``C_noisy(theta_t)``, shape ``(B,)``.
        value: Target ``C_ZNE(theta_t)``, shape ``(B,)``.
        gradient: Target ``g_ZNE(theta_t)``, shape ``(B, n)``.
        step: Target ``theta_{t+1} - theta_t``, shape ``(B, n)``.
    """

    theta: FloatArray
    history: FloatArray
    history_noisy: FloatArray
    noisy: FloatArray
    value: FloatArray
    gradient: FloatArray
    step: FloatArray

    def subset(self, rows: NDArray[np.int64] | slice) -> TrainingBatch:
        """Rows ``rows`` of every array."""
        return TrainingBatch(
            self.theta[rows],
            self.history[rows],
            self.history_noisy[rows],
            self.noisy[rows],
            self.value[rows],
            self.gradient[rows],
            self.step[rows],
        )

    def __len__(self) -> int:
        """Number of steps."""
        return int(self.theta.shape[0])


def build_batch(
    thetas: FloatArray,
    noisy: FloatArray,
    *,
    window: int,
    values: FloatArray | None = None,
    gradients: FloatArray | None = None,
    steps: FloatArray | None = None,
    rows: list[int] | None = None,
) -> TrainingBatch:
    """Windows of a trajectory: one row per ``t in rows`` (default all), padded at the start.

    Args:
        thetas: ``theta_0 .. theta_T``, shape ``(T + 1, n)`` or ``(T, n)``.
        noisy: ``C_noisy(theta_k)`` for the same rows.
        window: History length ``W``.
        values: Target values per row (NaN where unknown).
        gradients: Target gradients per row.
        steps: Target steps per row.
        rows: The steps ``t`` to include.
    """
    thetas = np.asarray(thetas, dtype=float)
    noisy = np.asarray(noisy, dtype=float)
    count, n = thetas.shape
    rows = list(range(count)) if rows is None else rows
    hist = np.empty((len(rows), window + 1, n))
    hist_c = np.empty((len(rows), window + 1))
    for i, t in enumerate(rows):
        idx = np.clip(np.arange(t - window, t + 1), 0, None)
        hist[i] = thetas[idx]
        hist_c[i] = noisy[idx]
    nan_v = np.full(len(rows), np.nan)
    nan_g = np.full((len(rows), n), np.nan)
    return TrainingBatch(
        theta=thetas[rows],
        history=hist,
        history_noisy=hist_c,
        noisy=noisy[rows],
        value=nan_v if values is None else np.asarray(values, float)[rows],
        gradient=nan_g if gradients is None else np.asarray(gradients, float)[rows],
        step=nan_g.copy() if steps is None else np.asarray(steps, float)[rows],
    )


class Surrogate(Protocol):
    """What the PIDN workflow needs from a surrogate."""

    def fit(self, batch: TrainingBatch, **kwargs: Any) -> dict[str, Any]:
        """Fit to a training batch; returns training diagnostics."""
        ...

    def predict(self, batch: TrainingBatch) -> tuple[FloatArray, FloatArray, FloatArray]:
        """``(C_hat, grad_theta C_hat, theta_hat_{t+1} - theta_t)`` per row."""
        ...


# -- normalisation ------------------------------------------------------------------------------


@dataclass(slots=True)
class Normaliser:
    """Fixed affine scalings, fitted once on the training phase."""

    theta_center: FloatArray
    theta_scale: float
    noisy_mean: float
    noisy_scale: float
    value_mean: float
    value_scale: float
    step_scale: float
    gradient_scale: float

    @classmethod
    def fit(cls, batch: TrainingBatch, features: str) -> Normaliser:
        """From the training rows."""
        theta_center = batch.history[0, 0].copy()
        spread = float(np.std(batch.theta - theta_center)) if len(batch) > 1 else 0.0
        theta_scale = 0.5 if features == "fourier" else max(spread, 0.05)
        gnorm = float(np.median(np.linalg.norm(batch.gradient, axis=1)))
        n = batch.theta.shape[1]
        gradient_scale = max(gnorm, 1e-6)
        step_norms = np.linalg.norm(batch.step, axis=1)
        step_scale = max(float(np.median(step_norms)) / math.sqrt(n), 1e-6)
        value_scale = max(float(np.std(batch.value)), gnorm * theta_scale / math.sqrt(n), 1e-3)
        noisy_scale = max(float(np.std(batch.noisy)), 1e-3, value_scale)
        return cls(
            theta_center=theta_center,
            theta_scale=theta_scale,
            noisy_mean=float(np.mean(batch.noisy)),
            noisy_scale=noisy_scale,
            value_mean=float(np.mean(batch.value)),
            value_scale=value_scale,
            step_scale=step_scale,
            gradient_scale=gradient_scale,
        )


# -- network ------------------------------------------------------------------------------------


@dataclass(slots=True)
class _Params:
    tensors: dict[str, Tensor] = field(default_factory=dict)

    def add(self, name: str, value: FloatArray) -> Tensor:
        tensor = Tensor(value, requires_grad=True)
        self.tensors[name] = tensor
        return tensor


class PidnNetwork:
    """The dual-branch, dual-head PIDN network (numpy autodiff).

    Args:
        config: Architecture.
        num_parameters: ``n``, the number of variational parameters.
        normaliser: Scalings fitted on the training phase.
        seed: Weight initialisation seed.
    """

    def __init__(
        self,
        config: PidnModelConfig,
        num_parameters: int,
        normaliser: Normaliser,
        seed: int,
    ) -> None:
        """Initialise the weights."""
        self.config = config
        self.n = num_parameters
        self.norm = normaliser
        self._rng = np.random.default_rng(seed)
        self.params = _Params()
        n = num_parameters
        feat = 2 * n if config.features == "fourier" else n
        use_noisy = config.noisy_hidden > 0
        self._act = _ACTIVATIONS[config.activation]
        width = 0
        if config.gru_hidden:
            gru_in = feat + n + (1 if use_noisy else 0)
            h = config.gru_hidden
            for gate in ("r", "z", "n"):
                self._linear(f"gru_x{gate}", gru_in, h)
                self._linear(f"gru_h{gate}", h, h, bias=gate == "n")
            width += h
        if use_noisy:
            self._linear("noisy", 1, config.noisy_hidden)
            width += config.noisy_hidden
        if config.direct_theta:
            width += feat
        if width == 0:
            msg = "the surrogate needs at least one input branch"
            raise ValueError(msg)
        previous = width
        for i, size in enumerate(config.trunk):
            self._linear(f"trunk{i}", previous, size)
            previous = size
        self._linear("value_head", previous, 1, gain=0.1)
        if config.step_head:
            self._linear("step_head", previous, n, gain=0.1)

    # -- parameters -------------------------------------------------------------------------

    def _linear(
        self, name: str, fan_in: int, fan_out: int, *, bias: bool = True, gain: float = 1.0
    ) -> None:
        limit = gain * self.config.init_scale * math.sqrt(6.0 / (fan_in + fan_out))
        self.params.add(f"{name}.w", self._rng.uniform(-limit, limit, (fan_in, fan_out)))
        if bias:
            self.params.add(f"{name}.b", np.zeros(fan_out))

    def parameters(self) -> list[Tensor]:
        """Trainable tensors in a fixed order."""
        return list(self.params.tensors.values())

    def get_state(self) -> list[FloatArray]:
        """Copies of the weights."""
        return [p.value.copy() for p in self.parameters()]

    def set_state(self, state: list[FloatArray]) -> None:
        """Restore weights from :meth:`get_state`."""
        for p, value in zip(self.parameters(), state, strict=True):
            p.value = value.copy()

    def _apply(self, name: str, x: Tensor) -> Tensor:
        p = self.params.tensors
        out = x @ p[f"{name}.w"]
        bias = p.get(f"{name}.b")
        return out if bias is None else out + bias

    # -- forward ----------------------------------------------------------------------------

    def _features(self, theta: Tensor) -> Tensor:
        if self.config.features == "fourier":
            return concat([cos(2.0 * theta), sin(2.0 * theta)], axis=-1)
        return (theta - Tensor(self.norm.theta_center)) * (1.0 / self.norm.theta_scale)

    def forward(self, batch: TrainingBatch, theta: Tensor) -> tuple[Tensor, Tensor | None]:
        """``(C_hat (B, 1), step_hat (B, n) or None)`` with ``theta`` the current point."""
        cfg = self.config
        norm = self.norm
        parts: list[Tensor] = []
        use_noisy = cfg.noisy_hidden > 0
        if cfg.gru_hidden:
            last = batch.history.shape[1] - 1
            h = Tensor(np.zeros((len(batch), cfg.gru_hidden)))
            for k in range(1, last + 1):
                current = theta if k == last else Tensor(batch.history[:, k, :])
                previous = Tensor(batch.history[:, k - 1, :])
                step = (current - previous) * (1.0 / norm.step_scale)
                inputs = [self._features(current), step]
                if use_noisy:
                    c = (batch.history_noisy[:, k] - norm.noisy_mean) / norm.noisy_scale
                    inputs.append(Tensor(c[:, None]))
                x = concat(inputs, axis=-1)
                r = sigmoid(self._apply("gru_xr", x) + self._apply("gru_hr", h))
                z = sigmoid(self._apply("gru_xz", x) + self._apply("gru_hz", h))
                cand = tanh(self._apply("gru_xn", x) + r * self._apply("gru_hn", h))
                h = (1.0 - z) * cand + z * h
            parts.append(h)
        if use_noisy:
            c = (batch.noisy - norm.noisy_mean) / norm.noisy_scale
            parts.append(self._act(self._apply("noisy", Tensor(c[:, None]))))
        if cfg.direct_theta:
            parts.append(self._features(theta))
        x = parts[0] if len(parts) == 1 else concat(parts, axis=-1)
        for i in range(len(cfg.trunk)):
            x = self._act(self._apply(f"trunk{i}", x))
        value = self._apply("value_head", x) * norm.value_scale + norm.value_mean
        step_hat = self._apply("step_head", x) * norm.step_scale if cfg.step_head else None
        return value, step_hat

    def value_and_gradient(
        self, batch: TrainingBatch, *, create_graph: bool
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """``C_hat``, ``grad_theta C_hat`` (per row) and the step head."""
        theta = Tensor(batch.theta, requires_grad=True)
        value, step = self.forward(batch, theta)
        (gradient,) = grad(value.sum(), [theta], create_graph=create_graph)
        return value, gradient, step

    def predict(self, batch: TrainingBatch) -> tuple[FloatArray, FloatArray, FloatArray]:
        """``(C_hat, grad C_hat, step)`` as arrays (no graph kept)."""
        value, gradient, step = self.value_and_gradient(batch, create_graph=False)
        with no_grad():
            step_arr = np.full_like(batch.theta, np.nan) if step is None else step.value.copy()
        return value.value[:, 0].copy(), gradient.value.copy(), step_arr


# -- baselines ----------------------------------------------------------------------------------


class QuadraticSurrogate:
    """``C(theta) = c + g.d + d.H.d / 2`` around the last training point, ridge-fitted.

    Unknowns ``c`` (1), ``g`` (n), symmetric ``H`` (n(n+1)/2). Every training step gives one
    value equation and ``n`` gradient equations ``g + H d_t = g_ZNE(theta_t)``; with few steps
    the system is underdetermined and the ridge term (on ``H``, relative to the data scale)
    picks the smallest curvature consistent with the data, much like a quasi-Newton update.
    """

    def __init__(self, ridge: float = 1e-2) -> None:
        """Create the (unfitted) model."""
        self.ridge = ridge
        self.center: FloatArray | None = None
        self.c = 0.0
        self.g: FloatArray | None = None
        self.h: FloatArray | None = None

    def fit(self, batch: TrainingBatch, **kwargs: Any) -> dict[str, Any]:
        """Weighted ridge least squares on values and gradients.

        Value and gradient equations are scaled to unit variance so neither dominates.
        """
        del kwargs
        theta, n = batch.theta, batch.theta.shape[1]
        self.center = theta[-1].copy()
        d = theta - self.center
        iu = np.triu_indices(n)
        m = iu[0].size
        rows, rhs = [], []
        v_scale = max(float(np.std(batch.value)), 1e-3)
        g_scale = max(float(np.median(np.linalg.norm(batch.gradient, axis=1))) / math.sqrt(n), 1e-6)
        for t in range(len(batch)):
            outer = np.outer(d[t], d[t])
            sym = np.where(iu[0] == iu[1], 0.5 * outer[iu], outer[iu])
            rows.append(np.concatenate([[1.0], d[t], sym]) / v_scale)
            rhs.append(batch.value[t] / v_scale)
            for i in range(n):
                hrow = np.zeros(m)
                for k, (a, b) in enumerate(zip(iu[0], iu[1], strict=True)):
                    if a == i:
                        hrow[k] = d[t, b]
                    elif b == i:
                        hrow[k] = d[t, a]
                grow = np.zeros(n)
                grow[i] = 1.0
                rows.append(np.concatenate([[0.0], grow, hrow]) / g_scale)
                rhs.append(batch.gradient[t, i] / g_scale)
        a_mat = np.vstack(rows)
        b_vec = np.asarray(rhs)
        penalty = np.zeros(1 + n + m)
        penalty[1 + n :] = self.ridge * np.sqrt(len(rhs))
        a_aug = np.vstack([a_mat, np.diag(penalty)])
        b_aug = np.concatenate([b_vec, np.zeros(1 + n + m)])
        solution, *_ = np.linalg.lstsq(a_aug, b_aug, rcond=None)
        self.c = float(solution[0])
        self.g = solution[1 : 1 + n]
        hess = np.zeros((n, n))
        hess[iu] = solution[1 + n :]
        self.h = hess + np.triu(hess, 1).T
        residual = a_mat @ solution - b_vec
        return {"residual_rms": float(np.sqrt(np.mean(residual**2))), "epochs": 0}

    def predict(self, batch: TrainingBatch) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Model value and gradient at every row; no step head."""
        assert self.center is not None
        assert self.g is not None
        assert self.h is not None
        d = batch.theta - self.center
        gradient = self.g[None, :] + d @ self.h
        value = self.c + d @ self.g + 0.5 * np.einsum("bi,ij,bj->b", d, self.h, d)
        return value, gradient, np.full_like(batch.theta, np.nan)


class LastGradientSurrogate:
    """Keeps the last training gradient and value: the trivial trajectory continuation."""

    def __init__(self) -> None:
        """Create the (unfitted) model."""
        self.gradient: FloatArray | None = None
        self.value = math.nan

    def fit(self, batch: TrainingBatch, **kwargs: Any) -> dict[str, Any]:
        """Remember the last row."""
        del kwargs
        self.gradient = batch.gradient[-1].copy()
        self.value = float(batch.value[-1])
        return {"epochs": 0}

    def predict(self, batch: TrainingBatch) -> tuple[FloatArray, FloatArray, FloatArray]:
        """The remembered gradient at every row."""
        assert self.gradient is not None
        rows = len(batch)
        return (
            np.full(rows, self.value),
            np.repeat(self.gradient[None, :], rows, axis=0),
            np.full_like(batch.theta, np.nan),
        )
