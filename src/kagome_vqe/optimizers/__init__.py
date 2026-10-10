"""Classical minimisers with a uniform interface.

All optimisers implement :class:`Optimizer`: ``minimize(objective, x0, callback=...)`` returning
an :class:`OptimizeResult`. Objectives are :class:`Objective` instances that can evaluate a
*batch* of parameter vectors at once (:meth:`Objective.batch`); the optimisers use this so that,
for example, one ADAM gradient step (``n + 1`` energies) becomes a single Estimator job on
hardware.

Evaluation budgets, for ``maxiter`` iterations and ``n`` parameters:

=========  ==========================  ==============================
optimiser  evaluations per iteration   total
=========  ==========================  ==============================
ADAM       ``n + 1`` (forward diff.)   ``maxiter * (n + 1)``
SPSA       ``2``                       ``2 * maxiter + 1`` (final)
scipy      method dependent            reported in ``nfev``
=========  ==========================  ==============================

ADAM and :class:`GradientDescent` accept any estimator from
:mod:`kagome_vqe.optimizers.gradients`: finite differences, simultaneous perturbation (the
SPSA-ADAM hybrid, 2 energies per iteration) or parameter shift (``2K`` energies for ``K``
parameterised gates).
"""

from kagome_vqe.optimizers.adam import Adam
from kagome_vqe.optimizers.base import (
    FunctionObjective,
    IterationInfo,
    Objective,
    Optimizer,
    OptimizeResult,
    as_objective,
)
from kagome_vqe.optimizers.gd import GradientDescent
from kagome_vqe.optimizers.gradients import (
    CentralDifference,
    ExpandableObjective,
    ForwardDifference,
    GradientEstimate,
    GradientEstimator,
    ParameterShift,
    SimultaneousPerturbation,
    make_gradient,
)
from kagome_vqe.optimizers.scipy import ScipyOptimizer
from kagome_vqe.optimizers.spsa import Spsa

__all__ = [
    "Adam",
    "CentralDifference",
    "ExpandableObjective",
    "ForwardDifference",
    "FunctionObjective",
    "GradientDescent",
    "GradientEstimate",
    "GradientEstimator",
    "IterationInfo",
    "Objective",
    "OptimizeResult",
    "Optimizer",
    "ParameterShift",
    "ScipyOptimizer",
    "SimultaneousPerturbation",
    "Spsa",
    "as_objective",
    "make_gradient",
]
