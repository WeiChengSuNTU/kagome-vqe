"""PIDN: the autodiff engine (to second order), the network, training, baselines, workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

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
from kagome_vqe.pidn.model import (
    LastGradientSurrogate,
    Normaliser,
    PidnModelConfig,
    PidnNetwork,
    QuadraticSurrogate,
    build_batch,
)
from kagome_vqe.pidn.training import PidnEnsemble, PidnTrainingConfig, pidn_losses

RING4 = {"kind": "custom", "num_sites": 4, "bonds": [[0, 1], [1, 2], [2, 3], [3, 0]]}


def _fd_grad(fn: Any, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    out = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        up, down = x.copy(), x.copy()
        up[idx] += eps
        down[idx] -= eps
        out[idx] = (fn(up) - fn(down)) / (2 * eps)
    return out


# -- autodiff -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op",
    [tanh, sigmoid, silu, sin, cos, lambda x: x * x * 0.5 - 2.0 * x, lambda x: (x @ x.T).sum(0)],
)
def test_first_and_second_derivatives_match_finite_differences(op: Any) -> None:
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(3, 3))
    w = rng.normal(size=(3, 3))

    def value(x: np.ndarray) -> float:
        return float((op(Tensor(x)) * Tensor(w)).sum().value)

    def first(x: np.ndarray) -> np.ndarray:
        t = Tensor(x, requires_grad=True)
        (g,) = grad((op(t) * Tensor(w)).sum(), [t])
        return g.value

    assert np.allclose(first(x0), _fd_grad(value, x0), atol=1e-7)
    # second order: d/dx of sum(c * dy/dx)
    c = rng.normal(size=(3, 3))

    def second_value(x: np.ndarray) -> float:
        return float(np.sum(c * first(x)))

    t = Tensor(x0, requires_grad=True)
    (g,) = grad((op(t) * Tensor(w)).sum(), [t], create_graph=True)
    (h,) = grad((g * Tensor(c)).sum(), [t])
    assert np.allclose(h.value, _fd_grad(second_value, x0), atol=1e-6)


def test_broadcasting_slicing_concat_and_reductions() -> None:
    rng = np.random.default_rng(1)
    a0, b0 = rng.normal(size=(4, 3)), rng.normal(size=(3,))

    def f(a: Tensor, b: Tensor) -> Tensor:
        joined = concat([a[:, :2] * b[:2], tanh(a + b)[:, 1:]], axis=1)
        return (joined.mean(axis=0) * 3.0).sum() + a.reshape((3, 4)).T.sum(axis=1).square().sum()

    a, b = Tensor(a0, requires_grad=True), Tensor(b0, requires_grad=True)
    ga, gb = grad(f(a, b), [a, b])
    assert np.allclose(ga.value, _fd_grad(lambda x: float(f(Tensor(x), Tensor(b0)).value), a0))
    assert np.allclose(gb.value, _fd_grad(lambda x: float(f(Tensor(a0), Tensor(x)).value), b0))


def test_relu_has_no_curvature_and_no_grad_records_nothing() -> None:
    x = Tensor(np.array([-1.0, 0.5, 2.0]), requires_grad=True)
    (g,) = grad(relu(x).square().sum(), [x], create_graph=True)
    assert np.allclose(g.value, [0.0, 1.0, 4.0])
    (h,) = grad((relu(x) * 1.0).sum(), [x], create_graph=True)
    (hh,) = grad(h.sum(), [x])
    assert np.allclose(hh.value, 0.0)
    with no_grad():
        y = tanh(x)
    assert not y.requires_grad


# -- network --------------------------------------------------------------------------------------


def _toy_batch(n: int = 5, steps: int = 9, window: int = 3) -> Any:
    rng = np.random.default_rng(3)
    thetas = np.cumsum(0.05 * rng.normal(size=(steps + 1, n)), axis=0)
    noisy = -10 + 0.1 * rng.normal(size=steps + 1)
    return build_batch(
        thetas[:steps],
        noisy[:steps],
        window=window,
        values=noisy[:steps] - 0.5,
        gradients=rng.normal(size=(steps, n)),
        steps=thetas[1:] - thetas[:-1],
    )


@pytest.mark.parametrize("features", ["raw", "fourier"])
@pytest.mark.parametrize("gru_hidden", [0, 4])
def test_network_input_gradient_is_exact(features: str, gru_hidden: int) -> None:
    batch = _toy_batch()
    cfg = PidnModelConfig(
        features=features,
        history=3,
        gru_hidden=gru_hidden,
        noisy_hidden=3,
        trunk=[6],
    )
    net = PidnNetwork(cfg, 5, Normaliser.fit(batch, features), seed=2)
    _, gradient, _ = net.predict(batch)

    def value_at(shift: float, i: int) -> np.ndarray:
        moved = _toy_batch()
        moved.theta = moved.theta.copy()
        moved.history = moved.history.copy()
        moved.theta[:, i] += shift
        moved.history[:, -1, i] += shift
        return net.predict(moved)[0]

    eps = 1e-6
    fd = np.column_stack([(value_at(eps, i) - value_at(-eps, i)) / (2 * eps) for i in range(5)])
    assert np.allclose(gradient, fd, atol=1e-7)


def test_physics_loss_weight_gradient_is_exact() -> None:
    batch = _toy_batch()
    cfg = PidnModelConfig(history=3, gru_hidden=3, noisy_hidden=2, trunk=[5])
    net = PidnNetwork(cfg, 5, Normaliser.fit(batch, "raw"), seed=4)
    training = PidnTrainingConfig(beta=2.0, step_weight=0.5)
    params = net.parameters()
    grads = grad(pidn_losses(net, batch, training, create_graph=True)["total"], params)
    for p, g in zip(params[:5], grads[:5], strict=True):
        idx = (0,) * p.value.ndim
        old = p.value.copy()
        p.value = old.copy()
        p.value[idx] += 1e-6
        up = float(pidn_losses(net, batch, training, create_graph=False)["total"].value)
        p.value = old.copy()
        p.value[idx] -= 1e-6
        down = float(pidn_losses(net, batch, training, create_graph=False)["total"].value)
        p.value = old
        assert g.value[idx] == pytest.approx((up - down) / 2e-6, abs=1e-6)


def test_training_fits_a_known_gradient_field() -> None:
    """On a quadratic landscape, the trained network's gradients point the right way."""
    rng = np.random.default_rng(5)
    n = 4
    hess = np.diag([1.0, 2.0, 0.5, 1.5])
    thetas = [rng.normal(size=n)]
    for _ in range(14):
        thetas.append(thetas[-1] - 0.1 * hess @ thetas[-1])
    thetas_arr = np.array(thetas)
    grads = thetas_arr[:-1] @ hess
    values = 0.5 * np.einsum("ti,ij,tj->t", thetas_arr[:-1], hess, thetas_arr[:-1])
    batch = build_batch(
        thetas_arr[:-1],
        values + 0.3,
        window=4,
        values=values,
        gradients=grads,
        steps=thetas_arr[1:] - thetas_arr[:-1],
    )
    model = PidnEnsemble(
        PidnModelConfig(history=4, gru_hidden=8, noisy_hidden=4, trunk=[32]),
        PidnTrainingConfig(
            epochs=600, ensemble=2, patience=200, learning_rate=1e-2, validation_fraction=0.2
        ),
    )
    info = model.fit(batch)
    _, predicted, _ = model.predict(batch)
    cos = np.sum(predicted * grads, axis=1) / (
        np.linalg.norm(predicted, axis=1) * np.linalg.norm(grads, axis=1)
    )
    assert cos.mean() > 0.95
    assert info["holdout_steps"] == 3
    assert len(info["epochs_selected"]) == 2
    assert model.spread(batch).shape == (len(batch),)


def test_quadratic_and_last_gradient_baselines() -> None:
    rng = np.random.default_rng(6)
    n = 3
    hess = np.array([[2.0, 0.3, 0.0], [0.3, 1.0, 0.1], [0.0, 0.1, 0.5]])
    thetas = rng.normal(size=(12, n))
    grads = thetas @ hess + 0.2
    values = 0.5 * np.einsum("ti,ij,tj->t", thetas, hess, thetas) + thetas @ np.full(n, 0.2)
    batch = build_batch(
        thetas, values, window=2, values=values, gradients=grads, steps=np.zeros_like(thetas)
    )
    quad = QuadraticSurrogate(ridge=1e-8)
    quad.fit(batch)
    _, g, _ = quad.predict(batch)
    assert np.allclose(g, grads, atol=1e-5)
    last = LastGradientSurrogate()
    last.fit(batch)
    assert np.allclose(last.predict(batch)[1], grads[-1])


def test_build_batch_pads_the_start_of_the_trajectory() -> None:
    thetas = np.arange(12, dtype=float).reshape(4, 3)
    batch = build_batch(thetas, np.arange(4.0), window=2, rows=[0, 3])
    assert batch.history.shape == (2, 3, 3)
    assert np.allclose(batch.history[0], thetas[[0, 0, 0]])
    assert np.allclose(batch.history[1], thetas[[1, 2, 3]])
    assert np.allclose(batch.history_noisy[1], [1, 2, 3])


# -- workflow -------------------------------------------------------------------------------------


def _zne_gd_source(tmp_path: Path) -> Path:
    from kagome_vqe.config import RunConfig
    from kagome_vqe.results import run_and_save

    cfg = RunConfig.model_validate(
        {
            "name": "zne-src",
            "seed": 2,
            "x0_scale": 0.3,
            "lattice": RING4,
            "ansatz": {"type": "graph_hva", "reps": 1},
            "track_ideal_gradient": True,
            "backend": {
                "kind": "logical_noise",
                "shots": 256,
                "final_shots": 512,
                "logical_noise": {
                    "channel": "depolarizing",
                    "probability": 0.01,
                    "fast_kernels": True,
                },
            },
            "mitigation": {
                "zne": {"scales": [1, 2], "extrapolation": "linear", "noise_scaling": "probability"}
            },
            "optimizer": {
                "type": "gd",
                "gradient": "parameter_shift",
                "lr": 0.05,
                "maxiter": 6,
                "measure_current": True,
            },
        }
    )
    return run_and_save(cfg, root=tmp_path / "src")


@pytest.mark.parametrize(
    "variant",
    [
        {"label": "pidn", "training": {"epochs": 60, "ensemble": 1, "patience": 20}},
        {
            "label": "pidn_anchor",
            "training": {"epochs": 40, "ensemble": 2, "refresh_epochs": 10},
            "stage3": {"anchor_every": 2, "max_step_factor": 2.0},
        },
        {
            "label": "pidn_noise_aware_anchor",
            "training": {"epochs": 40, "ensemble": 1, "refresh_epochs": 10, "noise_aware": True},
            "stage3": {"anchor_every": 2},
        },
        {"label": "quad", "model": {"kind": "quadratic"}, "stage3": {"measure_noisy": False}},
        {"label": "last", "model": {"kind": "last_gradient"}, "stage3": {"measure_noisy": False}},
    ],
)
def test_pidn_workflow_continues_a_zne_run_and_counts_costs(
    tmp_path: Path, variant: dict[str, Any]
) -> None:
    from kagome_vqe.pidn.workflow import PidnStudyConfig, run_pidn_study

    source = _zne_gd_source(tmp_path)
    variant = {"model": {"history": 2, "gru_hidden": 4, "noisy_hidden": 2, "trunk": [8]}} | variant
    cfg = PidnStudyConfig.model_validate(
        {
            "name": "pidn-test",
            "sources": [str(source)],
            "output_dir": str(tmp_path / "out"),
            "t_init": 4,
            "total_iterations": 8,
            "variants": [variant],
        }
    )
    study = run_pidn_study(cfg)
    (run_dir,) = sorted((study / "runs").glob("*"))
    result = json.loads((run_dir / "result.json").read_text())
    with np.load(source / "trajectory.npz") as src:
        stage1 = int(src["circuit_executions"][3])
    with np.load(run_dir / "trajectory.npz") as data:
        traj = {k: data[k] for k in data.files}
    assert traj["theta"].shape == (8, 4)
    assert (traj["phase"][:4] == 0).all()
    assert np.allclose(traj["theta"][4], traj["theta_next"][3])
    r = result["resources"]
    assert r["stage1_circuit_executions"] == stage1
    groups = 3
    measure = variant.get("stage3", {}).get("measure_noisy", True)
    assert r["stage3_circuit_executions"] == (4 * groups if measure else 0)
    anchors = r["anchors"]
    assert (
        r["circuit_executions"]
        == stage1 + r["stage3_circuit_executions"] + (r["anchor_circuit_executions"])
    )
    if variant["label"].endswith("anchor"):
        assert anchors == 2
        assert r["anchor_circuit_executions"] == 2 * (2 * 4 + 1) * groups * 2
        assert np.isfinite(traj["anchor_cosine"]).sum() == 2
    else:
        assert anchors == 0
    assert result["final_ideal_relative_error"] is not None
    assert np.isfinite(result["final_zne_energy"])
    assert np.all(np.isfinite(traj["ideal_grad"]))
    assert (study / "summary" / "report.md").exists()


def test_variant_validation() -> None:
    from kagome_vqe.pidn.workflow import PidnVariant

    with pytest.raises(ValueError, match="step_head"):
        PidnVariant.model_validate(
            {"label": "x", "model": {"kind": "quadratic"}, "stage3": {"update": "step_head"}}
        )
    with pytest.raises(ValueError, match="measure_noisy"):
        PidnVariant.model_validate({"label": "x", "stage3": {"measure_noisy": False}})


def test_open_loop_extrapolation_scores_every_variant() -> None:
    from kagome_vqe.pidn.diagnostics import extrapolation_test
    from kagome_vqe.pidn.workflow import PidnVariant

    rng = np.random.default_rng(8)
    n, steps = 3, 14
    hess = np.diag([1.0, 0.5, 2.0])
    theta = [rng.normal(size=n)]
    for _ in range(steps):
        theta.append(theta[-1] - 0.1 * hess @ theta[-1])
    th = np.array(theta)
    grads = th[:-1] @ hess
    traj = {
        "theta": th[:-1],
        "theta_next": th[1:],
        "value": 0.5 * np.einsum("ti,ij,tj->t", th[:-1], hess, th[:-1]),
        "grad": grads,
        "scale_values": np.zeros((steps, 1)),
        "ideal_grad": grads,
    }
    variants = [
        PidnVariant.model_validate(
            {
                "label": "last",
                "model": {"kind": "last_gradient"},
                "stage3": {"measure_noisy": False},
            }
        ),
        PidnVariant.model_validate(
            {
                "label": "quad",
                "model": {"kind": "quadratic", "quadratic_ridge": 1e-6},
                "stage3": {"measure_noisy": False},
            }
        ),
    ]
    frame = extrapolation_test(traj, variants, t_init=8)
    assert len(frame) == 2 * (steps - 8)
    quad = frame[frame["variant"] == "quad"]
    assert (quad["cosine_to_ideal"] > 0.999).all()  # exact for a quadratic landscape
    last = frame[frame["variant"] == "last"]
    assert last["ahead"].tolist() == list(range(1, steps - 8 + 1))


def test_pidn_continues_a_noise_free_run_at_no_circuit_cost(tmp_path: Path) -> None:
    from kagome_vqe.config import RunConfig
    from kagome_vqe.pidn.workflow import PidnStudyConfig, run_pidn_study
    from kagome_vqe.results import run_and_save

    source = run_and_save(
        RunConfig.model_validate(
            {
                "name": "sv-src",
                "seed": 3,
                "x0_scale": 0.3,
                "lattice": RING4,
                "ansatz": {"type": "graph_hva", "reps": 1},
                "track_ideal_gradient": True,
                "backend": {"kind": "statevector"},
                "optimizer": {
                    "type": "gd",
                    "gradient": "parameter_shift",
                    "lr": 0.05,
                    "maxiter": 5,
                    "measure_current": True,
                },
            }
        ),
        root=tmp_path / "src",
    )
    cfg = PidnStudyConfig.model_validate(
        {
            "name": "clean",
            "sources": [str(source)],
            "output_dir": str(tmp_path / "out"),
            "t_init": 4,
            "total_iterations": 7,
            "variants": [
                {
                    "label": "pidn",
                    "model": {"history": 2, "gru_hidden": 3, "noisy_hidden": 2, "trunk": [6]},
                    "training": {"epochs": 30, "ensemble": 1},
                },
            ],
        }
    )
    study = run_pidn_study(cfg)
    result = json.loads((study / "runs" / "pidn-s3" / "result.json").read_text())
    assert result["resources"]["circuit_executions"] == 0
    assert result["final_zne_energy"] == pytest.approx(result["final_ideal_energy"])
    with np.load(study / "runs" / "pidn-s3" / "trajectory.npz") as data:
        assert np.allclose(data["scale_values"][4:, 0], data["ideal_energy"][4:])


def test_relabelled_stage_one_feeds_the_pidn_workflow(tmp_path: Path) -> None:
    from kagome_vqe.pidn.relabel import RelabelConfig, run_relabel
    from kagome_vqe.pidn.workflow import PidnStudyConfig, run_pidn_study

    source = _zne_gd_source(tmp_path)
    backend = {
        "kind": "logical_noise",
        "shots": 256,
        "final_shots": 512,
        "logical_noise": {
            "channel": "depolarizing",
            "probability": 0.01,
            "readout_error": 0.02,
            "fast_kernels": True,
        },
    }
    relabel = RelabelConfig.model_validate(
        {
            "name": "relabel",
            "sources": [str(source)],
            "output_dir": str(tmp_path / "relabel"),
            "t_init": 4,
            "pipelines": [
                {
                    "label": "m2",
                    "backend": backend,
                    "mitigation": {"readout": True, "symmetry": True, "rescale": {"twin": "auto"}},
                },
            ],
        }
    )
    out = run_relabel(relabel)
    run_dir = out / "runs" / "m2-s2"
    with np.load(run_dir / "trajectory.npz") as data:
        traj = {k: data[k] for k in data.files}
    with np.load(source / "trajectory.npz") as data:
        original = {k: data[k] for k in data.files}
    assert np.allclose(traj["theta"], original["theta"])
    assert np.isfinite(traj["grad"][:4]).all()
    assert np.isnan(traj["grad"][4:]).all()
    assert np.isfinite(traj["scale_values"][:, 0]).all()
    assert traj["evaluations"][3] == 4 * (2 * 4 + 1)
    assert traj["circuit_executions"][3] == 4 * 9 * 3
    result = json.loads((run_dir / "result.json").read_text())
    assert len(result["gradient_cosine_to_ideal"]) == 4
    assert "readout_calibration" in json.dumps(result["estimator"])
    cfg = PidnStudyConfig.model_validate(
        {
            "name": "pidn-relabelled",
            "sources": [str(run_dir)],
            "output_dir": str(tmp_path / "pidn"),
            "t_init": 4,
            "total_iterations": 6,
            "baseline_studies": [],
            "variants": [
                {
                    "label": "last",
                    "model": {"kind": "last_gradient"},
                    "stage3": {"measure_noisy": False},
                }
            ],
        }
    )
    study = run_pidn_study(cfg)
    final = json.loads((study / "runs" / "last-m2-s104731" / "result.json").read_text())
    assert final["resources"]["stage1_circuit_executions"] == 4 * 9 * 3
    assert np.isfinite(final["final_zne_energy"])


def test_target_quality_measures_gradient_noise() -> None:
    from kagome_vqe.pidn.diagnostics import target_quality

    rng = np.random.default_rng(9)
    ideal = rng.normal(size=(10, 4))
    traj = {
        "grad": ideal + 0.1 * rng.normal(size=(10, 4)),
        "ideal_grad": ideal,
        "value": np.full(10, -1.0),
        "ideal_energy": np.full(10, -1.2),
    }
    q = target_quality(traj, t_init=10)
    assert q["cosine_to_ideal"] > 0.95
    assert q["value_error_mean"] == pytest.approx(0.2)
    assert q["signal_to_noise_late"] > 5


def test_stage_three_noisy_input_uses_the_stage_one_estimator() -> None:
    """C_noisy in Stage III must come from the estimator behind the Stage I noisy column."""
    from types import SimpleNamespace

    from kagome_vqe.config import MitigationConfig, ZneConfig
    from kagome_vqe.pidn.workflow import noisy_input_mitigation

    def run(mitigation: MitigationConfig, **result: Any) -> Any:
        return SimpleNamespace(config=SimpleNamespace(mitigation=mitigation), result=result)

    rs = MitigationConfig(readout=True, symmetry=True)
    # gradient descent without ZNE records the mitigated value itself
    assert noisy_input_mitigation(run(rs)) == rs
    # ZNE records the scale-1 energies of its inner (counts-level) stack
    zne = MitigationConfig(readout=True, zne=ZneConfig(scales=[1, 3, 5]))
    assert noisy_input_mitigation(run(zne)) == MitigationConfig(readout=True)
    assert not noisy_input_mitigation(run(MitigationConfig(zne=ZneConfig(scales=[1, 3])))).enabled
    # relabelled Stage I measures the raw cost explicitly
    assert not noisy_input_mitigation(run(rs, relabelled_from="x")).enabled


def test_noise_aware_loss_is_a_chi_square_and_stops_at_the_noise_floor() -> None:
    batch = _toy_batch()
    rng = np.random.default_rng(8)
    batch.gradient_variance = rng.uniform(0.5, 2.0, size=batch.gradient.shape)
    model_cfg = PidnModelConfig(history=3, gru_hidden=0, noisy_hidden=0, trunk=[16])
    plain_cfg = PidnTrainingConfig(epochs=400, ensemble=1)
    aware_cfg = plain_cfg.model_copy(update={"noise_aware": True, "noise_floor": 0.5})
    net = PidnNetwork(model_cfg, batch.theta.shape[1], Normaliser.fit(batch, "raw"), 0)
    _, predicted, _ = net.value_and_gradient(batch, create_graph=False)
    expected = np.mean((predicted.value - batch.gradient) ** 2 / batch.gradient_variance)
    got = float(pidn_losses(net, batch, aware_cfg, create_graph=False)["physics"].value)
    assert got == pytest.approx(expected)
    with pytest.raises(ValueError, match="variances"):
        pidn_losses(net, _toy_batch(), aware_cfg, create_graph=False)
    aware = PidnEnsemble(model_cfg, aware_cfg)
    info = aware.fit(batch)
    assert info["epochs_selected"][0] < 400  # stopped at the floor, not after every epoch
    assert info["final_physics_loss"] <= 0.5 + 1e-9
    assert batch.subset(slice(0, 3)).gradient_variance.shape == (3, batch.theta.shape[1])


def test_gradient_variances_follow_the_shift_rule_and_zne_weights() -> None:
    from kagome_vqe.ansatz.unshare import ParameterMap
    from kagome_vqe.pidn.workflow import gradient_variances

    mapping = ParameterMap(np.array([0, 0, 1, 2, 2, 2]), 3, None, np.full(6, 1.0))
    traj = {
        "zne_weights": np.array([[1.0], [1.0]]),
        "scale_stds": np.array([[0.1], [0.2]]),
    }
    var = gradient_variances(traj, mapping)
    assert np.allclose(var[0], 2 * 0.01 * np.array([2, 1, 3]))
    assert np.allclose(var[1], 4 * var[0])
    zne = {"zne_weights": np.array([[2.0, -1.0]]), "scale_stds": np.array([[0.1, 0.3]])}
    assert np.allclose(gradient_variances(zne, mapping)[0], 2 * (0.04 + 0.09) * np.array([2, 1, 3]))
