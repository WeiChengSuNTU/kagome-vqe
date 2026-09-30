"""Thin wrapper over :func:`scipy.optimize.minimize`."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from scipy import optimize

from kagome_vqe.optimizers.base import (
    Callback,
    FloatArray,
    IterationInfo,
    Objective,
    OptimizeResult,
    as_objective,
)

__all__ = ["ScipyOptimizer"]


class ScipyOptimizer:
    """Any gradient-free or numerical-gradient scipy method (COBYLA, Powell, L-BFGS-B, ...).

    scipy evaluates points one at a time, so nothing is batched; every evaluation is one job on
    hardware. Prefer :class:`~kagome_vqe.optimizers.Spsa` there.

    Args:
        method: scipy method name.
        maxiter: Iteration budget (passed through ``options["maxiter"]``).
        tol: scipy's ``tol``.
        options: Extra method options merged over ``{"maxiter": maxiter}``.
    """

    def __init__(
        self,
        method: str = "COBYLA",
        maxiter: int = 100,
        tol: float | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        """See class docstring."""
        self._method = method
        self._maxiter = maxiter
        self._tol = tol
        self._options = dict(options or {})

    @property
    def name(self) -> str:
        """``"scipy_<method>"`` in lower case."""
        return f"scipy_{self._method.lower()}"

    @property
    def maxiter(self) -> int:
        """Iteration budget."""
        return self._maxiter

    def evaluations_per_iteration(self, num_parameters: int) -> None:
        """Unknown; depends on the method."""
        return None

    def minimize(
        self,
        objective: Objective | Callable[[FloatArray], float],
        x0: FloatArray,
        *,
        callback: Callback | None = None,
    ) -> OptimizeResult:
        """Delegate to scipy; iteration callbacks are forwarded when the method supports them."""
        fn = as_objective(objective)
        nfev = 0
        nit = 0
        history: list[float] = []

        def wrapped(x: FloatArray) -> float:
            nonlocal nfev
            nfev += 1
            return fn(x)

        def on_iteration(intermediate: Any) -> None:
            nonlocal nit
            nit += 1
            x = np.asarray(getattr(intermediate, "x", intermediate), dtype=float)
            value = float(getattr(intermediate, "fun", np.nan))
            if np.isnan(value):
                value = fn(x)
            history.append(value)
            if callback is not None:
                callback(IterationInfo(nit=nit, x=x.copy(), fun=value, nfev=nfev))

        res = optimize.minimize(
            wrapped,
            np.array(x0, dtype=float),
            method=self._method,
            tol=self._tol,
            options={"maxiter": self._maxiter, **self._options},
            callback=on_iteration,
        )
        return OptimizeResult(
            x=np.asarray(res.x, dtype=float),
            fun=float(res.fun),
            nfev=nfev,
            nit=int(getattr(res, "nit", nit) or nit),
            success=bool(res.success),
            message=str(res.message),
            history=history,
        )

    def __repr__(self) -> str:
        """Debug representation."""
        return f"ScipyOptimizer(method={self._method!r}, maxiter={self._maxiter})"
