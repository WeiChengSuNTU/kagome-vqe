"""Error mitigation (roadmap milestone M2).

Two layers that compose freely:

* :mod:`~kagome_vqe.mitigation.counts` measures each basis itself and post-processes the shots:
  readout-error correction and total-spin symmetry filtering.
* :mod:`~kagome_vqe.mitigation.wrappers` acts on energies from any estimator: zero-noise
  extrapolation and twin-circuit rescaling.

:func:`kagome_vqe.estimators.make_estimator` assembles the stack from ``RunConfig.mitigation``:
counts-level processing innermost, then ZNE, then rescaling.
"""

from kagome_vqe.mitigation.counts import (
    CountsEnergy,
    MeasurementGroup,
    ReadoutCalibration,
    measurement_plan,
)
from kagome_vqe.mitigation.wrappers import (
    Extrapolation,
    TwinRescaledEnergy,
    ZneEnergy,
    choose_twin,
    extrapolate,
)

__all__ = [
    "CountsEnergy",
    "Extrapolation",
    "MeasurementGroup",
    "ReadoutCalibration",
    "TwinRescaledEnergy",
    "ZneEnergy",
    "choose_twin",
    "extrapolate",
    "measurement_plan",
]
