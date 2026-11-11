"""Optimiser selection, budgets and resource accounting, on cheap synthetic objectives."""

from __future__ import annotations

from types import TracebackType
from typing import Any, Self

import numpy as np
import pytest

from kagome_vqe.config import BackendConfig, BudgetConfig, OptimizerConfig, RunConfig
from kagome_vqe.estimators import EnergyBatch
from kagome_vqe.optimizers import Adam, ScipyOptimizer, SimultaneousPerturbation, Spsa
from kagome_vqe.vqe import BudgetExhaustedError, EnergyObjective, initial_point


class QuadraticEstimator:
    """Synthetic estimator: E(x) = |x - 1|^2, optionally reporting shots."""

    def __init__(self, shots: int | None = None) -> None:
        self.shots = shots
        self.calls = 0

    @property
    def description(self) -> dict[str, Any]:
        return {"kind": "quadratic"}

    def evaluate(self, params: np.ndarray, *, shots: int | None = None) -> EnergyBatch:
        params = np.atleast_2d(params)
        self.calls += params.shape[0]
        energies = np.sum((params - 1.0) ** 2, axis=1)
        return EnergyBatch(energies, np.zeros(len(energies)), shots or self.shots)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


# -- optimiser selection ----------------------------------------------------------------------


def test_adam_spsa_really_uses_simultaneous_perturbation() -> None:
    cfg = OptimizerConfig(type="adam", gradient="spsa", perturbation=0.1, lr=0.02)
    opt = cfg.build(np.random.default_rng(0))
    assert isinstance(opt, Adam)
    assert isinstance(opt.gradient, SimultaneousPerturbation)
    assert opt.gradient.perturbation == 0.1
    assert opt.evaluations_per_iteration(50) == 2
    assert cfg.label() == "adam_spsa"
    assert "simultaneous perturbation" in cfg.gradient_description()


def test_default_adam_is_forward_difference_and_says_so() -> None:
    cfg = OptimizerConfig(type="adam")
    opt = cfg.build(np.random.default_rng(0))
    assert isinstance(opt, Adam)
    assert opt.gradient.name == "forward"
    assert opt.evaluations_per_iteration(16) == 17
    assert cfg.label() == "adam"
    assert cfg.gradient_description().startswith("forward")


def test_spsa_and_scipy_selection_with_hyperparameters() -> None:
    spsa = OptimizerConfig(
        type="spsa", calibrate=True, target_magnitude=0.2, calibration_samples=7, alpha=0.7
    ).build(np.random.default_rng(0))
    assert isinstance(spsa, Spsa)
    assert spsa.calibration_cost == 14
    scipy = OptimizerConfig(type="scipy", method="L-BFGS-B", options={"eps": 0.1})
    assert isinstance(scipy.build(np.random.default_rng(0)), ScipyOptimizer)
    assert "eps = 0.1" in scipy.gradient_description()
    assert "derivative-free" in OptimizerConfig(type="scipy").gradient_description()


# -- evaluation accounting --------------------------------------------------------------------


def test_calibrated_spsa_counts_calibration_evaluations() -> None:
    est = QuadraticEstimator()
    objective = EnergyObjective(est)
    spsa = Spsa(maxiter=10, perturbation=0.1, calibrate=True, calibration_samples=5, seed=1)
    result = spsa.minimize(objective, np.zeros(3))
    assert spsa.calibration_evaluations == 10
    assert result.nfev == 10 + 2 * 10 + 1 == objective.nfev == est.calls


def test_scipy_callback_costs_no_extra_evaluation() -> None:
    """The iteration callback must not re-evaluate the objective (it used to, once per step)."""
    est = QuadraticEstimator()
    objective = EnergyObjective(est)
    result = ScipyOptimizer("L-BFGS-B", maxiter=50).minimize(objective, np.zeros(4))
    assert result.nfev == objective.nfev == est.calls
    assert len(result.history) == result.nit


def test_budget_refuses_batches_before_evaluating() -> None:
    est = QuadraticEstimator()
    objective = EnergyObjective(est)
    objective.set_budget(max_evaluations=5)
    objective.batch(np.zeros((4, 2)))
    with pytest.raises(BudgetExhaustedError):
        objective.batch(np.zeros((2, 2)))  # 4 + 2 > 5: refused whole
    assert est.calls == objective.nfev == 4
    objective(np.zeros(2))  # exactly 5 is allowed
    with pytest.raises(BudgetExhaustedError):
        objective(np.zeros(2))


def test_shot_budget_uses_total_cost() -> None:
    est = QuadraticEstimator(shots=100)
    objective = EnergyObjective(est)
    objective.set_budget(
        max_shots=1000, shots_per_evaluation=100, shot_cost=lambda measured: 3 * measured
    )
    objective.batch(np.zeros((3, 2)))  # 3 x 100 x 3 bases = 900
    with pytest.raises(BudgetExhaustedError, match="max_shots"):
        objective(np.zeros(2))  # would be 1200
    assert objective.shots_spent == 300


# -- configuration ----------------------------------------------------------------------------


def test_budget_config_validation_and_round_trip() -> None:
    cfg = RunConfig(budget=BudgetConfig(max_evaluations=10), backend=BackendConfig())
    assert RunConfig.model_validate(cfg.to_toml_dict()) == cfg
    with pytest.raises(ValueError, match="max_shots"):
        RunConfig(budget=BudgetConfig(max_shots=10), backend=BackendConfig(kind="statevector"))
    assert not RunConfig().budget.enabled


def test_initial_point_depends_only_on_seed_scale_and_size() -> None:
    a = RunConfig(seed=7, x0_scale=0.1, optimizer=OptimizerConfig(type="spsa"))
    b = RunConfig(seed=7, x0_scale=0.1, optimizer=OptimizerConfig(type="adam", gradient="spsa"))
    np.testing.assert_array_equal(initial_point(a, 16), initial_point(b, 16))
    assert not np.array_equal(
        initial_point(a, 16), initial_point(a.model_copy(update={"seed": 8}), 16)
    )
    assert np.all((initial_point(a, 16) >= 0) & (initial_point(a, 16) < 0.1))
