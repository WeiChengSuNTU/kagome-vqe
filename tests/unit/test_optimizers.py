import numpy as np
import pytest

from kagome_vqe.optimizers import (
    Adam,
    FunctionObjective,
    IterationInfo,
    Objective,
    Optimizer,
    ScipyOptimizer,
    Spsa,
    as_objective,
)

TARGET = np.array([1.5, -0.5, 2.0])


def quadratic(x: np.ndarray) -> float:
    return float(np.sum((x - TARGET) ** 2))


class CountingObjective:
    """Records batch sizes so tests can check that iterations are batched."""

    def __init__(self) -> None:
        self.batches: list[int] = []

    def __call__(self, x: np.ndarray) -> float:
        self.batches.append(1)
        return quadratic(x)

    def batch(self, xs: np.ndarray) -> np.ndarray:
        self.batches.append(xs.shape[0])
        return np.array([quadratic(x) for x in xs])


def test_as_objective_wraps_plain_callables() -> None:
    obj = as_objective(quadratic)
    assert isinstance(obj, FunctionObjective)
    assert isinstance(obj, Objective)
    assert obj.batch(np.zeros((2, 3))).shape == (2,)
    assert as_objective(obj) is obj


@pytest.mark.parametrize(
    "optimizer",
    [
        Adam(maxiter=400, lr=0.05),
        Adam(maxiter=400, lr=0.05, gradient="central", amsgrad=True),
        Spsa(maxiter=400, learning_rate=0.1, perturbation=0.05, seed=0),
        Spsa(maxiter=150, calibrate=True, perturbation=0.05, seed=0),
        ScipyOptimizer("COBYLA", maxiter=300),
        ScipyOptimizer("Powell", maxiter=50),
    ],
)
def test_converges_on_quadratic(optimizer: Optimizer) -> None:
    assert isinstance(optimizer, Optimizer)
    res = optimizer.minimize(quadratic, np.zeros(3))
    assert res.x == pytest.approx(TARGET, abs=0.05)
    assert res.fun < 5e-3
    assert res.nfev > 0
    assert res.nit >= 1


def test_adam_evaluation_accounting() -> None:
    obj = CountingObjective()
    opt = Adam(maxiter=7, lr=0.1, tol=0.0)
    res = opt.minimize(obj, np.zeros(3))
    assert opt.evaluations_per_iteration(3) == 4
    assert res.nit == 7
    assert res.nfev == 7 * 4 + 1  # gradient batches plus a final evaluation
    assert obj.batches[:7] == [4] * 7  # each iteration is one batch of n + 1 points


def test_adam_central_accounting() -> None:
    opt = Adam(maxiter=3, lr=0.1, gradient="central", tol=0.0)
    res = opt.minimize(CountingObjective(), np.zeros(3))
    assert opt.evaluations_per_iteration(3) == 7
    assert res.nfev == 3 * 7 + 1


def test_spsa_evaluation_accounting() -> None:
    obj = CountingObjective()
    opt = Spsa(maxiter=10, seed=3)
    res = opt.minimize(obj, np.zeros(3))
    assert opt.evaluations_per_iteration(3) == 2
    assert res.nfev == 2 * 10 + 1
    assert obj.batches[:10] == [2] * 10


def test_spsa_is_deterministic_with_seed() -> None:
    a = Spsa(maxiter=20, seed=7).minimize(quadratic, np.zeros(3))
    b = Spsa(maxiter=20, seed=7).minimize(quadratic, np.zeros(3))
    assert a.x == pytest.approx(b.x)


def test_callbacks_report_iterations() -> None:
    seen: list[IterationInfo] = []
    Adam(maxiter=5, lr=0.1, tol=0.0).minimize(quadratic, np.zeros(3), callback=seen.append)
    assert [i.nit for i in seen] == [1, 2, 3, 4, 5]
    assert seen[-1].nfev == 20
    seen.clear()
    Spsa(maxiter=4, seed=1).minimize(quadratic, np.zeros(3), callback=seen.append)
    assert [i.nit for i in seen] == [1, 2, 3, 4]


def test_adam_stops_on_tolerance() -> None:
    res = Adam(maxiter=1000, lr=0.1, tol=1e-3).minimize(quadratic, np.zeros(3))
    assert res.nit < 1000
    assert "tol" in res.message


def test_invalid_settings() -> None:
    with pytest.raises(ValueError, match="maxiter"):
        Adam(maxiter=0)
    with pytest.raises(ValueError, match="gradient"):
        Adam(gradient="sideways")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="perturbation"):
        Spsa(perturbation=0)
