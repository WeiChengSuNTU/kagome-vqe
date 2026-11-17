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
        Adam(gradient="sideways")
    with pytest.raises(ValueError, match="perturbation"):
        Spsa(perturbation=0)


# -- model gradient descent and momentum -------------------------------------------------------


def test_sample_ball_stays_inside_the_radius() -> None:
    from kagome_vqe.optimizers.mgd import sample_ball

    center = np.array([0.3, -1.0, 2.0])
    pts = sample_ball(np.random.default_rng(0), center, 0.2, 500)
    dist = np.linalg.norm(pts - center, axis=1)
    assert pts.shape == (500, 3)
    assert dist.max() <= 0.2 + 1e-12
    assert dist.mean() > 0.12  # uniform in the ball: most mass near the surface (3/4 r)


def test_quadratic_model_gradient_is_exact_for_a_quadratic() -> None:
    from kagome_vqe.optimizers.mgd import quadratic_model_gradient, sample_ball

    rng = np.random.default_rng(1)
    a = rng.normal(size=(4, 4))
    hess = a @ a.T
    b = rng.normal(size=4)

    def f(x: np.ndarray) -> float:
        return float(0.5 * x @ hess @ x + b @ x + 3.0)

    center = np.array([0.1, 0.2, -0.3, 0.4])
    xs = sample_ball(rng, center, 0.3, 40)  # 15 coefficients: overdetermined
    value, grad = quadratic_model_gradient(xs, np.array([f(x) for x in xs]), center)
    assert value == pytest.approx(f(center))
    assert np.allclose(grad, hess @ center + b)


def test_mgd_minimises_a_noisy_quadratic_and_records_iterations() -> None:
    from kagome_vqe.optimizers import ModelGradientDescent

    rng = np.random.default_rng(2)
    target = np.array([0.5, -0.4, 0.2])

    def f(x: np.ndarray) -> float:
        return float(np.sum((x - target) ** 2) + 0.01 * rng.normal())

    opt = ModelGradientDescent(maxiter=60, lr=0.2, sample_radius=0.3, sample_points=12, seed=3)
    res = opt.minimize(f, np.zeros(3))
    assert np.linalg.norm(res.x - target) < 0.1
    assert res.nfev == 60 * 12 + 1
    assert len(opt.records) == 60
    assert opt.records[-1]["details"]["region_points"] >= 12


def test_gradient_descent_momentum_is_the_heavy_ball_rule() -> None:
    from kagome_vqe.optimizers import GradientDescent

    def f(x: np.ndarray) -> float:
        return float(x @ x)

    exact = _ExactGradient()
    plain = GradientDescent(maxiter=2, lr=0.1, gradient=exact).minimize(f, np.array([1.0]))
    heavy = GradientDescent(maxiter=2, lr=0.1, gradient=exact, momentum=0.5).minimize(
        f, np.array([1.0])
    )
    # plain: 1 -> 0.8 -> 0.64; heavy ball: v1 = 2, x1 = 0.8; v2 = 0.5*2 + 1.6 = 2.6, x2 = 0.54
    assert plain.x[0] == pytest.approx(0.64)
    assert heavy.x[0] == pytest.approx(0.54)
    with pytest.raises(ValueError, match="momentum"):
        GradientDescent(momentum=1.0)


class _ExactGradient:
    name = "exact"

    def evaluations(self, num_parameters: int) -> int:
        del num_parameters
        return 0

    def estimate(self, fn: object, x: np.ndarray) -> object:
        from kagome_vqe.optimizers import GradientEstimate

        del fn
        return GradientEstimate(grad=2 * x, value=None, nfev=0)


def test_mgd_and_momentum_configs() -> None:
    from kagome_vqe.config import OptimizerConfig
    from kagome_vqe.optimizers import GradientDescent, ModelGradientDescent

    mgd = OptimizerConfig(type="mgd", lr=0.1, sample_radius=0.2, sample_points=20, maxiter=5)
    built = mgd.build(np.random.default_rng(0))
    assert isinstance(built, ModelGradientDescent)
    assert built.evaluations_per_iteration(7) == 20
    assert "20 new energies" in mgd.gradient_description()
    gdm = OptimizerConfig(type="gd", gradient="parameter_shift", momentum=0.9)
    assert isinstance(gdm.build(np.random.default_rng(0)), GradientDescent)
    with pytest.raises(ValueError, match="momentum"):
        OptimizerConfig(type="adam", momentum=0.9)
