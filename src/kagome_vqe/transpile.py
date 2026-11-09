"""Compile a logical (circuit, observable) pair into an ISA problem for a backend.

The important idiom, and the one the original notebook got wrong, is that after transpilation
the observable must be re-expressed on the *physical* qubits the transpiler chose:
``observable.apply_layout(isa_circuit.layout)``. With that, any backend can be used and the
energy is the same as on the logical circuit (up to noise).

:func:`compile_mapping` does the same for a circuit that a hardware-mapping strategy
(:mod:`kagome_vqe.mapping`) has already routed: the routed grid is placed on hardware qubits by
subgraph matching, transpiled with routing disabled (so the strategy's SWAPs are the only ones),
and the observable follows the strategy's final permutation and the transpiler's layout.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager

if TYPE_CHECKING:
    from kagome_vqe.mapping import MappingResult

__all__ = ["IsaProblem", "compile_for_backend", "compile_mapping", "fold_two_qubit_gates"]

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
        logical_observable: The Hamiltonian on the ansatz (logical) qubits, before layout.
            Counts-level estimators use it with ``physical_qubits`` to decide what to measure.
        two_qubit_depth: Depth counting only two-qubit gates.
        mapping: Description of the hardware-mapping strategy, placement and resources, or
            ``None`` for the plain transpiler path.
    """

    circuit: QuantumCircuit
    observable: SparsePauliOp
    backend_name: str = "logical"
    physical_qubits: tuple[int, ...] | None = None
    optimization_level: int | None = None
    gate_counts: dict[str, int] = field(default_factory=dict)
    depth: int = 0
    two_qubit_gates: int = 0
    logical_observable: SparsePauliOp | None = None
    two_qubit_depth: int = 0
    mapping: dict[str, Any] | None = None

    @property
    def measurement_qubits(self) -> tuple[int, ...]:
        """Physical qubit hosting each logical qubit (identity for logical problems)."""
        if self.physical_qubits is not None:
            return self.physical_qubits
        n = (
            self.observable.num_qubits
            if self.logical_observable is None
            else self.logical_observable.num_qubits
        )
        return tuple(range(n))

    def with_circuit(self, circuit: QuantumCircuit) -> IsaProblem:
        """Same problem with a different circuit on the same qubits (e.g. noise-folded)."""
        counts, depth, two_q = _describe(circuit)
        return replace(
            self,
            circuit=circuit,
            gate_counts=counts,
            depth=depth,
            two_qubit_gates=two_q,
            two_qubit_depth=_two_qubit_depth(circuit),
        )

    @property
    def measurement_groups(self) -> int:
        """Measurement bases needed for one energy: qubit-wise commuting groups of the observable.

        Three for the Heisenberg model (all-X, all-Y, all-Z). One energy evaluation at ``N``
        shots therefore costs ``measurement_groups * N`` circuit executions.
        """
        return len(self.observable.group_commuting(qubit_wise=True))

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
            "two_qubit_depth": self.two_qubit_depth,
            "gate_counts": dict(self.gate_counts),
            "measurement_groups": self.measurement_groups,
            "mapping": self.mapping,
        }


def _two_qubit_depth(circuit: QuantumCircuit) -> int:
    return int(circuit.depth(lambda inst: inst.operation.num_qubits == 2))


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
            logical_observable=observable,
            two_qubit_depth=_two_qubit_depth(logical),
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
        logical_observable=observable,
        two_qubit_depth=_two_qubit_depth(isa_circuit),
    )


def _has_coupling(backend: BackendV2) -> bool:
    try:
        return backend.coupling_map is not None
    except (AttributeError, NotImplementedError):
        return False


def compile_mapping(
    result: MappingResult,
    logical_observable: SparsePauliOp,
    backend: BackendV2 | None,
    *,
    optimization_level: int = 3,
    initial_layout: Sequence[int] | None = None,
    physical_layout: Sequence[int] | None = None,
    seed_transpiler: int | None = None,
) -> IsaProblem:
    """Turn a :class:`~kagome_vqe.mapping.MappingResult` into an ISA problem.

    Args:
        result: Output of :func:`kagome_vqe.mapping.map_circuit`.
        logical_observable: Hamiltonian on the logical qubits.
        backend: Target, or ``None`` to evaluate the (routed) circuit as is.
        optimization_level: Preset pass-manager level for the native-gate compilation.
        initial_layout: Generic strategy only: physical qubit per logical qubit.
        physical_layout: Geometric strategies only: hardware qubit per routed-circuit qubit
            (validated against the coupling map); ``None`` searches with calibration data.
        seed_transpiler: Seed for stochastic passes.

    Returns:
        The ISA problem; ``physical_qubits`` gives the qubit hosting each *logical* qubit when
        it is measured, and ``mapping`` records strategy, status, placement and resources.

    Raises:
        kagome_vqe.mapping.MappingError: If the routed grid cannot be placed on ``backend``.
    """
    description = result.describe()
    if not result.is_routed:
        problem = compile_for_backend(
            result.circuit,
            logical_observable,
            backend,
            optimization_level=optimization_level,
            initial_layout=initial_layout,
            seed_transpiler=seed_transpiler,
        )
        return replace(problem, mapping=description)

    if backend is None:
        counts, depth, two_q = _describe(result.circuit)
        return IsaProblem(
            circuit=result.circuit.copy(),
            observable=result.observable,
            physical_qubits=tuple(result.final_qubits),
            gate_counts=counts,
            depth=depth,
            two_qubit_gates=two_q,
            logical_observable=logical_observable,
            two_qubit_depth=_two_qubit_depth(result.circuit),
            mapping=description,
        )

    from kagome_vqe.mapping.placement import place_on_backend

    width = result.circuit.num_qubits
    layout: list[int] | None
    if _has_coupling(backend):
        usage: Counter[tuple[int, int]] = Counter()
        for inst in result.circuit.data:
            if inst.operation.num_qubits == 2:
                a, b = (result.circuit.find_bit(q).index for q in inst.qubits)
                usage[(min(a, b), max(a, b))] += 1
        placement = place_on_backend(
            width,
            sorted(result.coupling),
            backend,
            usage=usage,
            measured=result.final_qubits,
            explicit=physical_layout,
        )
        layout = list(placement.physical)
        description.update(placement.describe())
    else:  # e.g. an ideal AerSimulator: no connectivity constraint
        layout = list(range(width))
        description.update({"physical_qubits": layout, "placement_method": "unconstrained"})
    pass_manager = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend,
        initial_layout=layout,
        routing_method="none",
        seed_transpiler=seed_transpiler,
    )
    isa_circuit: QuantumCircuit = pass_manager.run(result.circuit)
    isa_layout = isa_circuit.layout
    isa_observable = result.observable.apply_layout(isa_layout)
    final_index = (
        list(range(isa_circuit.num_qubits))
        if isa_layout is None
        else [int(q) for q in isa_layout.final_index_layout()]
    )
    physical = tuple(final_index[c] for c in result.final_qubits)
    counts, depth, two_q = _describe(isa_circuit)
    log.info(
        "mapped %s (%s, %s) onto %s: depth=%d, 2q gates=%d, 2q depth=%d",
        result.used,
        result.status,
        f"{width} qubits",
        backend.name,
        depth,
        two_q,
        _two_qubit_depth(isa_circuit),
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
        logical_observable=logical_observable,
        two_qubit_depth=_two_qubit_depth(isa_circuit),
        mapping=description,
    )


def fold_two_qubit_gates(circuit: QuantumCircuit, scale: int) -> QuantumCircuit:
    """Amplify two-qubit noise by local folding: two-qubit ``G`` becomes ``G (G^dag G)^k``.

    With ``scale = 2k + 1`` the circuit implements the same unitary but each two-qubit gate is
    applied ``scale`` times, so its error is amplified roughly ``scale``-fold. Single-qubit
    gates are left alone: two-qubit gates dominate the noise. The folded circuit must not be
    re-optimised, or the transpiler would cancel the inserted pairs.

    Args:
        circuit: An ISA circuit.
        scale: Odd noise-scale factor, 1 returns a copy.

    Raises:
        ValueError: If ``scale`` is not a positive odd integer.
    """
    if scale < 1 or scale % 2 == 0:
        msg = f"fold scale must be a positive odd integer, got {scale}"
        raise ValueError(msg)
    folded = circuit.copy_empty_like()
    pairs = (scale - 1) // 2
    for inst in circuit.data:
        folded.append(inst.operation, inst.qubits, inst.clbits)
        if inst.operation.num_qubits == 2 and inst.operation.name not in ("barrier", "measure"):
            inverse = inst.operation.inverse()
            for _ in range(pairs):
                folded.append(inverse, inst.qubits, inst.clbits)
                folded.append(inst.operation, inst.qubits, inst.clbits)
    return folded
