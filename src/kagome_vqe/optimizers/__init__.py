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
from kagome_vqe.optimizers.scipy import ScipyOptimizer
from kagome_vqe.optimizers.spsa import Spsa

__all__ = [
    "Adam",
    "FunctionObjective",
    "IterationInfo",
    "Objective",
    "OptimizeResult",
    "Optimizer",
    "ScipyOptimizer",
    "Spsa",
    "as_objective",
]
