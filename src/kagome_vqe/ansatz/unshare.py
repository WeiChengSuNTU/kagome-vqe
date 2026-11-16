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

Exchange gates
--------------
The Heisenberg exchange gate of ``graph_hva`` and ``hva``,

    G(theta) = exp(-i theta/2 (XX + YY + ZZ)) = RZZ(theta) RYY(theta) RXX(theta),

is three commuting rotations driven by one angle. Its generator ``(XX + YY + ZZ)/2`` has only
two distinct eigenvalues, ``+1/2`` (triplet, three-fold) and ``-3/2`` (singlet): a gap of 2.
For a generator with two eigenvalues a gap ``Delta`` apart, every expectation value is
``a + b cos(Delta theta) + c sin(Delta theta)`` in that angle, so

    dE/dtheta = Delta / (2 sin(Delta s)) [E(theta + s) - E(theta - s)]

for any shift ``s``. With ``Delta = 2`` and ``s = pi/4`` this is ``E(theta + pi/4) -
E(theta - pi/4)``: two energies per exchange gate instead of six for its three rotations.
``unshare_parameters(..., exchange_gates=True)`` treats each such triple as one occurrence and
records the rule in :attr:`ParameterMap.shifts` and :attr:`ParameterMap.coefficients`. The rule
is exact for any channel that does not depend on the angles, so it holds for the PRB 106 layer
noise as well (only the shot noise remains).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from qiskit.circuit import Gate, Parameter, QuantumCircuit

from kagome_vqe.ansatz.gates import eswap

__all__ = [
    "EXCHANGE_COEFFICIENT",
    "EXCHANGE_SHIFT",
    "PAULI_COEFFICIENT",
    "PAULI_SHIFT",
    "SHIFTABLE_GATES",
    "ParameterMap",
    "unshare_parameters",
]

PAULI_SHIFT: Final[float] = float(np.pi / 2)
"""Shift of the two-term rule for ``exp(-i phi P/2)`` (generator gap 1)."""
PAULI_COEFFICIENT: Final[float] = 0.5
"""``dE/dphi = 0.5 [E(phi + pi/2) - E(phi - pi/2)]``."""
EXCHANGE_SHIFT: Final[float] = float(np.pi / 4)
"""Shift of the two-term rule for the exchange gate ``exp(-i theta/2 (XX+YY+ZZ))`` (gap 2)."""
EXCHANGE_COEFFICIENT: Final[float] = 1.0
"""``dE/dtheta = E(theta + pi/4) - E(theta - pi/4)``."""

_EXCHANGE_TRIPLE: Final[frozenset[str]] = frozenset({"rxx", "ryy", "rzz"})

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
        shifts: Per occurrence, the shift ``s`` of its two-term rule; ``None`` means ``pi/2``
            everywhere (Pauli rotations).
        coefficients: Per occurrence, the factor ``r`` in ``dE/dphi = r [E(phi + s) -
            E(phi - s)]``; ``None`` means ``1/2`` everywhere.
    """

    occurrence_to_param: IntArray
    num_parameters: int
    shifts: FloatArray | None = None
    coefficients: FloatArray | None = None

    @property
    def occurrence_shifts(self) -> FloatArray:
        """Shift per occurrence (``pi/2`` unless an exchange rule was recorded)."""
        if self.shifts is None:
            return np.full(self.num_occurrences, PAULI_SHIFT)
        return np.asarray(self.shifts, dtype=float)

    @property
    def occurrence_coefficients(self) -> FloatArray:
        """Shift-rule factor per occurrence (``1/2`` unless an exchange rule was recorded)."""
        if self.coefficients is None:
            return np.full(self.num_occurrences, PAULI_COEFFICIENT)
        return np.asarray(self.coefficients, dtype=float)

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


def _exchange_triple(circuit: QuantumCircuit, index: int) -> Parameter | None:
    """The shared parameter if ``circuit.data[index:index + 3]`` is one exchange gate."""
    window = circuit.data[index : index + 3]
    if len(window) < 3 or {inst.operation.name for inst in window} != _EXCHANGE_TRIPLE:
        return None
    qubit_sets = {frozenset(inst.qubits) for inst in window}
    params = {inst.operation.params[0] for inst in window}
    if len(qubit_sets) != 1 or len(params) != 1:
        return None
    (param,) = params
    return param if isinstance(param, Parameter) else None


def unshare_parameters(
    circuit: QuantumCircuit, *, exchange_gates: bool = False
) -> tuple[QuantumCircuit, ParameterMap]:
    """Rewrite ``circuit`` with one fresh parameter per parameter occurrence.

    Occurrence parameters are named ``occ0000``, ``occ0001``, ... in circuit order, so Qiskit's
    alphabetical parameter ordering equals occurrence order.

    Args:
        circuit: A parameterised circuit whose parameters enter only as the bare angle of gates
            in :data:`SHIFTABLE_GATES`. Gates without free parameters pass through unchanged.
        exchange_gates: Treat three consecutive ``rzz``, ``ryy``, ``rxx`` on the same pair with
            the same parameter as one exchange-gate occurrence (shift ``pi/4``, factor 1; see
            the module docs). ``False`` keeps the historical one-occurrence-per-rotation
            behaviour.

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
    shifts: list[float] = []
    coefficients: list[float] = []
    skip = 0
    for position, inst in enumerate(circuit.data):
        if skip:
            skip -= 1
            continue
        operation = inst.operation
        free = [p for p in operation.params if getattr(p, "parameters", None)]
        if not free:
            rewritten.append(operation, inst.qubits, inst.clbits)
            continue
        shared = _exchange_triple(circuit, position) if exchange_gates else None
        if shared is not None:
            new_param = Parameter(f"occ{len(occurrences):04d}")
            occurrences.append(index_of[shared])
            shifts.append(EXCHANGE_SHIFT)
            coefficients.append(EXCHANGE_COEFFICIENT)
            for member in circuit.data[position : position + 3]:
                rebuilt = type(member.operation)(new_param)
                rewritten.append(rebuilt, member.qubits, member.clbits)
            skip = 2
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
        shifts.append(PAULI_SHIFT)
        coefficients.append(PAULI_COEFFICIENT)
        rewritten.append(_rebuilt(operation, new_param), inst.qubits, inst.clbits)
    exchange = exchange_gates and any(c != PAULI_COEFFICIENT for c in coefficients)
    mapping = ParameterMap(
        np.asarray(occurrences, dtype=np.int64),
        len(original),
        shifts=np.asarray(shifts, dtype=float) if exchange else None,
        coefficients=np.asarray(coefficients, dtype=float) if exchange else None,
    )
    return rewritten, mapping
