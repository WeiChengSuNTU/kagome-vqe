"""The :class:`Ansatz` protocol and the shared "ring" skeleton of RVB and HVA."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from qiskit.circuit import Parameter, QuantumCircuit

from kagome_vqe.ansatz.gates import singlet_pair

__all__ = ["Ansatz", "RingAnsatz", "ring_bonds"]


@runtime_checkable
class Ansatz(Protocol):
    """A factory for parameterised circuits.

    Implementations must be cheap to construct and ``build()`` must return a *new* circuit on
    every call so callers can transpile or mutate it freely.
    """

    @property
    def name(self) -> str:
        """Short identifier used in file names and plots (e.g. ``"hva"``)."""
        ...

    @property
    def num_qubits(self) -> int:
        """Number of qubits the circuit acts on."""
        ...

    @property
    def num_parameters(self) -> int:
        """Number of free parameters in the built circuit."""
        ...

    def build(self) -> QuantumCircuit:
        """Construct the circuit."""
        ...


def ring_bonds(num_qubits: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Split the ring ``0-1-2-...-(n-1)-0`` into odd and even bonds.

    Reproduces the original notebook's ordering exactly: odd bonds are visited from the
    wrap-around bond ``(n-1, 0)`` downwards, even bonds from ``(0, 1)`` upwards.

    Args:
        num_qubits: Ring length; must be even so that the even bonds form a perfect matching.

    Returns:
        ``(odd_bonds, even_bonds)`` where ``odd_bonds`` are ``(n-1, 0), (n-3, n-2), ...``
        and ``even_bonds`` are ``(0, 1), (2, 3), ...``.
    """
    if num_qubits % 2:
        msg = f"ring ansatzes need an even number of qubits, got {num_qubits}"
        raise ValueError(msg)
    odd = [(i, 0) if i == num_qubits - 1 else (i, i + 1) for i in range(num_qubits - 1, 0, -2)]
    even = [(i, i + 1) for i in range(0, num_qubits, 2)]
    return odd, even


class RingAnsatz(ABC):
    """Common skeleton: a singlet product state followed by ``reps`` layers on the ring.

    Subclasses implement :meth:`_layer`, which appends one repetition's gates on the odd and
    even ring bonds, and :meth:`_layer_parameter_count`.

    The initial state places a singlet ``(|01> - |10>)/sqrt(2)`` on every even bond
    ``(2k, 2k+1)``. For the kagome lattice with the default Hamiltonian-cycle site order this
    is already a dimer covering with energy ``-3 * n / 2`` and the exact ground state is
    reached when all gates act trivially. See ``docs/physics.md``.
    """

    def __init__(self, num_qubits: int = 12, reps: int = 1) -> None:
        """Create a ring ansatz.

        Args:
            num_qubits: Ring length (even).
            reps: Number of gate layers.
        """
        if reps < 1:
            msg = f"reps must be >= 1, got {reps}"
            raise ValueError(msg)
        self._num_qubits = num_qubits
        self._reps = reps
        self._odd, self._even = ring_bonds(num_qubits)

    @property
    def num_qubits(self) -> int:
        """Number of qubits."""
        return self._num_qubits

    @property
    def reps(self) -> int:
        """Number of layers."""
        return self._reps

    @property
    def odd_bonds(self) -> list[tuple[int, int]]:
        """Ring bonds ``(n-1, 0), (n-3, n-2), ...``."""
        return list(self._odd)

    @property
    def even_bonds(self) -> list[tuple[int, int]]:
        """Ring bonds ``(0, 1), (2, 3), ...``; these also carry the initial singlets."""
        return list(self._even)

    @property
    def num_parameters(self) -> int:
        """``reps`` times the per-layer parameter count."""
        return self._reps * self._layer_parameter_count()

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier."""

    @abstractmethod
    def _layer_parameter_count(self) -> int:
        """Parameters introduced by one call to :meth:`_layer`."""

    @abstractmethod
    def _layer(self, circuit: QuantumCircuit, rep: int) -> None:
        """Append one layer to ``circuit`` in place; ``rep`` indexes the layer."""

    def build(self) -> QuantumCircuit:
        """Construct the circuit: singlets, then ``reps`` layers."""
        circuit = QuantumCircuit(self._num_qubits, name=self.name)
        for a, b in self._even:
            singlet_pair(circuit, a, b)
        for rep in range(self._reps):
            self._layer(circuit, rep)
        return circuit

    @staticmethod
    def _param(prefix: str, rep: int, index: int | None = None) -> Parameter:
        """Deterministically named parameter, zero-padded so alphabetical == creation order."""
        suffix = f"{rep:02d}" if index is None else f"{rep:02d}_{index:02d}"
        return Parameter(f"{prefix}{suffix}")

    def __repr__(self) -> str:
        """Debug representation."""
        return f"{type(self).__name__}(num_qubits={self._num_qubits}, reps={self._reps})"
