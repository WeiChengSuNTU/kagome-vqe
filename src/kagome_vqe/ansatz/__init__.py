"""Parameterised trial circuits.

Every ansatz implements the :class:`Ansatz` protocol: a ``name``, a parameter count, and
``build()`` returning a fresh :class:`~qiskit.circuit.QuantumCircuit` whose parameters are
ordered as ``circuit.parameters`` (alphabetically by name, which is how Qiskit orders them and
how the estimators bind values).

Two physically motivated ansatzes from the paper are provided:

* :class:`RVBAnsatz` -- resonating valence bond: singlet pairs followed by a ring of eSWAP
  gates, one parameter per gate.
* :class:`HVAnsatz` -- Hamiltonian variational ansatz: singlet pairs followed by Trotterised
  odd-bond and even-bond Heisenberg layers, one parameter per layer.

and, for arbitrary interaction graphs,

* :class:`GraphHVAnsatz` -- graph-based HVA: singlets on a dimer covering followed by an
  edge-coloured schedule of exchange gates, one parameter per gate, optional H0 evolution.

plus thin wrappers over Qiskit's hardware-efficient library circuits in
:mod:`kagome_vqe.ansatz.library`.
"""

from kagome_vqe.ansatz.base import Ansatz, RingAnsatz, ring_bonds
from kagome_vqe.ansatz.gates import eswap, legacy_eswap, singlet_pair
from kagome_vqe.ansatz.graph_hva import GateSpec, GraphHVAnsatz
from kagome_vqe.ansatz.hva import HVAnsatz, heisenberg_bond
from kagome_vqe.ansatz.library import LibraryAnsatz
from kagome_vqe.ansatz.rvb import RVBAnsatz

__all__ = [
    "Ansatz",
    "GateSpec",
    "GraphHVAnsatz",
    "HVAnsatz",
    "LibraryAnsatz",
    "RVBAnsatz",
    "RingAnsatz",
    "eswap",
    "heisenberg_bond",
    "legacy_eswap",
    "ring_bonds",
    "singlet_pair",
]
