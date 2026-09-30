"""Hardware-efficient ansatzes from Qiskit's circuit library, behind the :class:`Ansatz` API."""

from __future__ import annotations

from typing import Literal

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import efficient_su2, real_amplitudes

__all__ = ["LibraryAnsatz"]

LibraryKind = Literal["real_amplitudes", "efficient_su2"]


class LibraryAnsatz:
    """``real_amplitudes`` or ``efficient_su2`` with the challenge notebook's settings.

    These are problem-agnostic baselines: they start from ``|0...0>`` (not a singlet product)
    and their parameter count grows linearly with qubits and layers. The original notebook
    tried them with ``entanglement="full"`` before settling on RVB/HVA.

    Args:
        kind: Which library circuit.
        num_qubits: Width.
        reps: Number of entangling layers.
        entanglement: Qiskit entanglement pattern (``"full"``, ``"linear"``,
            ``"reverse_linear"``, ``"circular"``, ...).
        skip_final_rotation_layer: Drop the trailing rotation layer (as the notebook did).
    """

    def __init__(
        self,
        kind: LibraryKind,
        num_qubits: int = 12,
        reps: int = 1,
        entanglement: str = "reverse_linear",
        *,
        skip_final_rotation_layer: bool = True,
    ) -> None:
        """See class docstring."""
        if kind not in ("real_amplitudes", "efficient_su2"):
            msg = f"unknown library ansatz {kind!r}"
            raise ValueError(msg)
        self._kind: LibraryKind = kind
        self._num_qubits = num_qubits
        self._reps = reps
        self._entanglement = entanglement
        self._skip_final = skip_final_rotation_layer

    @property
    def name(self) -> str:
        """The library function name."""
        return self._kind

    @property
    def num_qubits(self) -> int:
        """Width."""
        return self._num_qubits

    @property
    def num_parameters(self) -> int:
        """Parameter count of the built circuit."""
        return self.build().num_parameters

    def build(self) -> QuantumCircuit:
        """Construct the circuit."""
        factory = real_amplitudes if self._kind == "real_amplitudes" else efficient_su2
        circuit: QuantumCircuit = factory(
            self._num_qubits,
            entanglement=self._entanglement,
            reps=self._reps,
            skip_final_rotation_layer=self._skip_final,
            name=self._kind,
        )
        return circuit

    def __repr__(self) -> str:
        """Debug representation."""
        return (
            f"LibraryAnsatz({self._kind!r}, num_qubits={self._num_qubits}, reps={self._reps}, "
            f"entanglement={self._entanglement!r})"
        )
