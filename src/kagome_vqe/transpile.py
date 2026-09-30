"""Compile a logical (circuit, observable) pair into an ISA problem for a backend.

The important idiom, and the one the original notebook got wrong, is that after transpilation
the observable must be re-expressed on the *physical* qubits the transpiler chose:
``observable.apply_layout(isa_circuit.layout)``. With that, any backend can be used and the
energy is the same as on the logical circuit (up to noise).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager

__all__ = ["IsaProblem", "compile_for_backend"]

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IsaProblem:
    """A circuit and observable ready for a specific backend.

    Attributes:
        circuit: The (possibly transpiled) circuit. Parameters keep their original order.
        observable: The Hamiltonian on the same qubits as ``circuit``.
        backend_name: Target device name, or ``"logical"`` when nothing was transpiled.
        physical_qubits: Physical qubit hosting each ansatz qubit, or ``None`` when logical.
        optimization_level: Preset level used, or ``None`` when logical.
        gate_counts: ``circuit.count_ops()`` as a plain dict.
        depth: Circuit depth.
        two_qubit_gates: Number of two-qubit gates (the dominant noise source).
    """

    circuit: QuantumCircuit
    observable: SparsePauliOp
    backend_name: str = "logical"
    physical_qubits: tuple[int, ...] | None = None
    optimization_level: int | None = None
    gate_counts: dict[str, int] = field(default_factory=dict)
    depth: int = 0
    two_qubit_gates: int = 0

    def stats(self) -> dict[str, object]:
        """Summary for logs and result files."""
        return {
            "backend": self.backend_name,
            "num_qubits": self.circuit.num_qubits,
            "num_parameters": self.circuit.num_parameters,
            "physical_qubits": None if self.physical_qubits is None else list(self.physical_qubits),
            "optimization_level": self.optimization_level,
            "depth": self.depth,
            "two_qubit_gates": self.two_qubit_gates,
            "gate_counts": dict(self.gate_counts),
        }


def _describe(circuit: QuantumCircuit) -> tuple[dict[str, int], int, int]:
    counts = {str(k): int(v) for k, v in circuit.count_ops().items()}
    two_q = sum(1 for inst in circuit.data if inst.operation.num_qubits == 2)
    return counts, int(circuit.depth()), two_q


def compile_for_backend(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    backend: BackendV2 | None,
    *,
    optimization_level: int = 3,
    initial_layout: Sequence[int] | None = None,
    seed_transpiler: int | None = None,
) -> IsaProblem:
    """Transpile ``circuit`` for ``backend`` and move ``observable`` onto the chosen qubits.

    Args:
        circuit: Logical ansatz circuit.
        observable: Hamiltonian on the logical qubits.
        backend: Target. ``None`` returns the inputs untouched (statevector evaluation), except
            that composite gates are decomposed so the statevector simulator can run them.
        optimization_level: Qiskit preset pass-manager level (3 = heaviest optimisation).
        initial_layout: Physical qubit for each logical qubit; ``None`` lets the layout pass
            choose (VF2/SABRE).
        seed_transpiler: Seed for the stochastic layout/routing passes.

    Returns:
        The ISA problem. ``observable`` has ``backend.num_qubits`` qubits with identities on
        the unused ones.
    """
    if backend is None:
        logical = circuit.decompose(reps=2)
        counts, depth, two_q = _describe(logical)
        return IsaProblem(
            circuit=logical,
            observable=observable,
            gate_counts=counts,
            depth=depth,
            two_qubit_gates=two_q,
        )

    pass_manager = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend,
        initial_layout=None if initial_layout is None else list(initial_layout),
        seed_transpiler=seed_transpiler,
    )
    isa_circuit: QuantumCircuit = pass_manager.run(circuit)
    layout = isa_circuit.layout
    isa_observable = observable.apply_layout(layout)
    physical = None if layout is None else tuple(int(q) for q in layout.final_index_layout())
    counts, depth, two_q = _describe(isa_circuit)
    log.info(
        "transpiled %s for %s: depth=%d, 2q gates=%d, physical qubits=%s",
        circuit.name,
        backend.name,
        depth,
        two_q,
        physical,
    )
    return IsaProblem(
        circuit=isa_circuit,
        observable=isa_observable,
        backend_name=str(backend.name),
        physical_qubits=physical,
        optimization_level=optimization_level,
        gate_counts=counts,
        depth=depth,
        two_qubit_gates=two_q,
    )
