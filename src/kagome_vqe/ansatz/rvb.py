"""Resonating valence bond (RVB) ansatz."""

from __future__ import annotations

from typing import Literal

from qiskit.circuit import QuantumCircuit

from kagome_vqe.ansatz.base import RingAnsatz
from kagome_vqe.ansatz.gates import eswap, legacy_eswap

__all__ = ["RVBAnsatz"]

ESwapKind = Literal["pauli_evolution", "legacy"]


class RVBAnsatz(RingAnsatz):
    """Singlet pairs plus a ring of independently parameterised eSWAP gates.

    Each layer applies an eSWAP on every odd ring bond, then on every even ring bond, with a
    fresh parameter per gate: ``num_qubits`` parameters per layer (12 for the kagome cell).
    Following Seki, Shirakawa and Yunoki (PRA 101, 052340), eSWAPs on a dimer state mix in the
    neighbouring dimer coverings, i.e. they build a resonating valence bond state.

    Args:
        num_qubits: Ring length (even).
        reps: Number of layers.
        gate: ``"pauli_evolution"`` for the true ``exp(-i phi/2 SWAP)`` (default) or
            ``"legacy"`` for the original notebook's gate, which carries an extra ``X``.
    """

    def __init__(
        self, num_qubits: int = 12, reps: int = 1, gate: ESwapKind = "pauli_evolution"
    ) -> None:
        """See class docstring."""
        super().__init__(num_qubits, reps)
        if gate not in ("pauli_evolution", "legacy"):
            msg = f"unknown eSWAP kind {gate!r}"
            raise ValueError(msg)
        self._gate = gate

    @property
    def name(self) -> str:
        """``"rvb"`` (or ``"rvb_legacy"``)."""
        return "rvb" if self._gate == "pauli_evolution" else "rvb_legacy"

    def _layer_parameter_count(self) -> int:
        return self._num_qubits

    def _layer(self, circuit: QuantumCircuit, rep: int) -> None:
        make = eswap if self._gate == "pauli_evolution" else legacy_eswap
        for index, (a, b) in enumerate([*self._odd, *self._even]):
            circuit.append(make(self._param("phi", rep, index)), [a, b])
