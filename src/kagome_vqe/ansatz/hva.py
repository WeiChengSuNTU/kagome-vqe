"""Hamiltonian variational ansatz (HVA)."""

from __future__ import annotations

from collections.abc import Sequence

from qiskit.circuit import ParameterExpression, QuantumCircuit

from kagome_vqe.ansatz.base import RingAnsatz

__all__ = ["HVAnsatz"]


def _heisenberg_bond(circuit: QuantumCircuit, theta: ParameterExpression, a: int, b: int) -> None:
    """Append ``exp(-i theta/2 (ZZ + YY + XX))`` on ``(a, b)`` as three commuting rotations."""
    circuit.rzz(theta, a, b)
    circuit.ryy(theta, a, b)
    circuit.rxx(theta, a, b)


class HVAnsatz(RingAnsatz):
    """Trotterised Heisenberg evolution on the odd and even ring bonds.

    One layer is ``exp(-i gamma/2 H_even) exp(-i phi/2 H_odd)`` where ``H_odd`` / ``H_even``
    are the Heisenberg terms ``XX + YY + ZZ`` on the odd / even ring bonds; the factor 1/2
    comes from Qiskit's rotation convention ``RZZ(theta) = exp(-i theta/2 ZZ)``. Two parameters
    per layer, shared across all bonds of a sub-layer, following Wiersema et al. (PRX Quantum 1,
    020319). Because the initial singlets already minimise ``H_even``, the state at all-zero
    parameters is a dimer covering, and for the kagome cell with the default site order it is
    the exact ground state.

    Args:
        num_qubits: Ring length (even).
        reps: Number of layers.
        extra_bonds: Optional additional bonds (in qubit labels) that get a third sub-layer
            with its own parameter ``kappa``. Use this to include the lattice bonds that are
            not on the ring (the six hexagon bonds of the kagome cell), which makes the ansatz
            lattice-aware at the cost of extra two-qubit depth.
    """

    def __init__(
        self,
        num_qubits: int = 12,
        reps: int = 1,
        extra_bonds: Sequence[tuple[int, int]] = (),
    ) -> None:
        """See class docstring."""
        super().__init__(num_qubits, reps)
        self._extra = [(int(a), int(b)) for a, b in extra_bonds]
        for a, b in self._extra:
            if not (0 <= a < num_qubits and 0 <= b < num_qubits) or a == b:
                msg = f"invalid extra bond ({a}, {b}) for {num_qubits} qubits"
                raise ValueError(msg)

    @property
    def name(self) -> str:
        """``"hva"`` (or ``"hva_lattice"`` when extra bonds are present)."""
        return "hva_lattice" if self._extra else "hva"

    @property
    def extra_bonds(self) -> list[tuple[int, int]]:
        """Bonds of the optional third sub-layer."""
        return list(self._extra)

    def _layer_parameter_count(self) -> int:
        return 3 if self._extra else 2

    def _layer(self, circuit: QuantumCircuit, rep: int) -> None:
        phi = self._param("phi", rep)
        gamma = self._param("gamma", rep)
        for a, b in self._odd:
            _heisenberg_bond(circuit, phi, a, b)
        for a, b in self._even:
            _heisenberg_bond(circuit, gamma, a, b)
        if self._extra:
            kappa = self._param("kappa", rep)
            for a, b in self._extra:
                _heisenberg_bond(circuit, kappa, a, b)
