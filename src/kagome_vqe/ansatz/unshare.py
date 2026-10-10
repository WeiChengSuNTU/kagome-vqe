"""Give every parameter occurrence its own parameter, for parameter-shift gradients.

The two-term parameter-shift rule

    dE/dphi = [ E(phi + pi/2) - E(phi - pi/2) ] / 2

is exact for a gate ``exp(-i phi G)`` whose generator ``G`` has eigenvalues ``+-1/2``. When
one ansatz parameter ``theta`` drives several such gates, as in HVA where each parameter
appears in 18 rotations, the rule must be applied to each *occurrence* separately and the
results summed (chain rule):

    dE/dtheta_i = sum over occurrences k of theta_i  of  dE/dphi_k

:func:`unshare_parameters` rewrites a circuit so that occurrence ``k`` carries its own
parameter ``phi_k``, and returns a :class:`ParameterMap` relating the two parameter vectors:
``phi = theta[occurrence_to_param]``. The rewritten circuit prepares exactly the same state at
``phi = expand(theta)``, so it can replace the original everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from qiskit.circuit import Gate, Parameter, QuantumCircuit

from kagome_vqe.ansatz.gates import eswap

__all__ = ["SHIFTABLE_GATES", "ParameterMap", "unshare_parameters"]

SHIFTABLE_GATES: Final[frozenset[str]] = frozenset(
    {"rx", "ry", "rz", "rxx", "ryy", "rzz", "rzx", "p", "eswap"}
)
"""Gates whose single angle enters as ``exp(-i angle G)`` with ``G`` of eigenvalues ``+-1/2``.

``p`` (phase) has eigenvalues ``0, 1``, which is ``+-1/2`` up to a global phase. ``eswap`` is
``exp(-i phi/2 SWAP)``; SWAP has eigenvalues ``+-1``. Controlled rotations (``crx``, used by the
legacy eSWAP) have three distinct eigenvalues and need a four-term rule: not supported.
"""

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ParameterMap:
    """Relation between ansatz parameters ``theta`` (n) and circuit parameters ``phi`` (K).

    Attributes:
        occurrence_to_param: Shape ``(K,)``; ``phi_k = theta[occurrence_to_param[k]]``.
        num_parameters: ``n``, the number of ansatz parameters.
    """

    occurrence_to_param: IntArray
    num_parameters: int

    @classmethod
    def identity(cls, num_parameters: int) -> ParameterMap:
        """The trivial map: every parameter occurs once, ``phi = theta``."""
        return cls(np.arange(num_parameters, dtype=np.int64), num_parameters)

    @property
    def num_occurrences(self) -> int:
        """``K``, the number of circuit parameters."""
        return int(self.occurrence_to_param.size)

    @property
    def is_identity(self) -> bool:
        """Whether ``phi`` and ``theta`` coincide."""
        return self.num_occurrences == self.num_parameters and bool(
            np.array_equal(self.occurrence_to_param, np.arange(self.num_parameters))
        )

    def expand(self, thetas: FloatArray) -> FloatArray:
        """Map ansatz parameters to circuit parameters; works for ``(n,)`` or ``(k, n)``."""
        arr = np.asarray(thetas, dtype=float)
        return np.asarray(arr[..., self.occurrence_to_param], dtype=float)

    def contract(self, grad_phi: FloatArray) -> FloatArray:
        """Chain rule: sum the per-occurrence derivatives belonging to each ansatz parameter."""
        return np.bincount(
            self.occurrence_to_param,
            weights=np.asarray(grad_phi, dtype=float),
            minlength=self.num_parameters,
        ).astype(float)


def _rebuilt(operation: Gate, new_param: Parameter) -> Gate:
    if operation.name == "eswap":
        return eswap(new_param)
    gate: Gate = type(operation)(new_param)
    return gate


def unshare_parameters(circuit: QuantumCircuit) -> tuple[QuantumCircuit, ParameterMap]:
    """Rewrite ``circuit`` with one fresh parameter per parameter occurrence.

    Occurrence parameters are named ``occ0000``, ``occ0001``, ... in circuit order, so Qiskit's
    alphabetical parameter ordering equals occurrence order.

    Args:
        circuit: A parameterised circuit whose parameters enter only as the bare angle of gates
            in :data:`SHIFTABLE_GATES`. Gates without free parameters pass through unchanged.

    Returns:
        The rewritten circuit and the map back to ``circuit.parameters``.

    Raises:
        ValueError: If a parameter enters through an unsupported gate or as an expression
            such as ``-phi / 2``, where the two-term rule does not apply.
    """
    original = list(circuit.parameters)
    index_of = {param: i for i, param in enumerate(original)}
    rewritten = circuit.copy_empty_like()
    occurrences: list[int] = []
    for inst in circuit.data:
        operation = inst.operation
        free = [p for p in operation.params if getattr(p, "parameters", None)]
        if not free:
            rewritten.append(operation, inst.qubits, inst.clbits)
            continue
        param = operation.params[0] if len(operation.params) == 1 else None
        if operation.name not in SHIFTABLE_GATES or not isinstance(param, Parameter):
            msg = (
                f"parameter shift needs each parameter to be the bare angle of one of "
                f"{sorted(SHIFTABLE_GATES)}; got {operation.name}({operation.params})"
            )
            raise ValueError(msg)
        new_param = Parameter(f"occ{len(occurrences):04d}")
        occurrences.append(index_of[param])
        rewritten.append(_rebuilt(operation, new_param), inst.qubits, inst.clbits)
    mapping = ParameterMap(np.asarray(occurrences, dtype=np.int64), len(original))
    return rewritten, mapping
