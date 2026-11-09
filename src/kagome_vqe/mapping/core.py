"""Strategy-independent machinery: logical blocks, the swap router, synthesis, verification.

Pipeline
--------
1. :func:`logical_ops` splits a logical circuit into one-qubit ops and two-qubit *blocks*
   (consecutive ``rxx``/``ryy``/``rzz`` on one pair become one ``exchange`` block, so a
   Heisenberg gate is routed and synthesised as a unit).
2. :class:`SwapRouter` places every logical qubit at a *home* position of an abstract square
   grid and walks the blocks **in circuit order**. Before a block whose qubits are not grid
   neighbours, it asks the strategy for candidate SWAPs (:class:`MoveRules`) that make them
   neighbours, undoing earlier SWAPs that are in the way. SWAPs are undone lazily, only when a
   later block needs the original arrangement, and SWAPs between two qubits nobody has touched
   yet (both still ``|0>``) are elided. Because blocks keep their order, the routed circuit
   implements the logical unitary exactly, up to the recorded final permutation.
3. :func:`synthesise` turns the routed op list into a :class:`~qiskit.QuantumCircuit` on the
   used grid positions; exchange blocks become a 3-CNOT circuit, and a SWAP directly adjacent
   to an exchange block on the same pair is optionally merged into it
   (``G(theta) SWAP = G(theta + pi/2)`` up to a global phase).

Auxiliary positions (swapping stations, empty grid sites) host qubits that start in ``|0>``,
are moved around by SWAPs but never acted on otherwise, so they stay ``|0>`` throughout.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

__all__ = [
    "EXCHANGE_GATES",
    "LogicalOp",
    "MappingError",
    "MoveRules",
    "PhysOp",
    "Position",
    "RoutedCircuit",
    "SwapRouter",
    "exchange_cx3",
    "logical_ops",
    "routed_observable",
    "synthesise",
    "verify_routed",
]

Position = tuple[int, int]
type Angle = Any
"""A gate angle: a float or a Qiskit ``ParameterExpression`` (Qiskit is untyped)."""
EXCHANGE_GATES = frozenset({"rxx", "ryy", "rzz"})
"""Gate names that make up a Heisenberg exchange block."""


class MappingError(RuntimeError):
    """A strategy cannot implement the requested circuit on its geometry."""


# -- logical ops ------------------------------------------------------------------------------


@dataclass(slots=True)
class LogicalOp:
    """One unit of the logical circuit.

    Attributes:
        qubits: Logical qubits (one or two).
        kind: ``"one"`` (single-qubit instruction), ``"exchange"`` (an
            ``exp(-i/2 (ax XX + ay YY + az ZZ))`` block) or ``"two"`` (any other two-qubit
            instruction).
        instructions: ``(operation, local qubit indices)`` to replay for ``one``/``two``.
        angles: ``(ax, ay, az)`` for ``exchange``.
    """

    qubits: tuple[int, ...]
    kind: Literal["one", "two", "exchange"]
    instructions: list[tuple[Any, tuple[int, ...]]] = field(default_factory=list)
    angles: list[Angle] = field(default_factory=lambda: [0.0, 0.0, 0.0])


def logical_ops(circuit: QuantumCircuit) -> list[LogicalOp]:
    """Split ``circuit`` into single-qubit ops and two-qubit blocks, preserving order.

    Raises:
        MappingError: On instructions with classical bits or more than two qubits.
    """
    ops: list[LogicalOp] = []
    last: dict[int, int] = {}  # qubit -> index in ops of its latest op
    axis = {"rxx": 0, "ryy": 1, "rzz": 2}
    for inst in circuit.data:
        operation = inst.operation
        if operation.name == "barrier":
            continue
        if inst.clbits:
            msg = f"cannot map instruction {operation.name} with classical bits"
            raise MappingError(msg)
        qubits = tuple(circuit.find_bit(q).index for q in inst.qubits)
        if len(qubits) == 1:
            ops.append(LogicalOp(qubits, "one", [(operation, (0,))]))
            last[qubits[0]] = len(ops) - 1
            continue
        if len(qubits) != 2:
            msg = f"cannot map {len(qubits)}-qubit instruction {operation.name}"
            raise MappingError(msg)
        a, b = qubits
        prev = last.get(a)
        is_exchange = operation.name in EXCHANGE_GATES
        if (
            is_exchange
            and prev is not None
            and prev == last.get(b)
            and ops[prev].kind == "exchange"
            and set(ops[prev].qubits) == {a, b}
        ):
            ops[prev].angles[axis[operation.name]] = (
                ops[prev].angles[axis[operation.name]] + operation.params[0]
            )
            continue
        if is_exchange:
            angles: list[Angle] = [0.0, 0.0, 0.0]
            angles[axis[operation.name]] = operation.params[0]
            ops.append(LogicalOp((a, b), "exchange", angles=angles))
        else:
            ops.append(LogicalOp((a, b), "two", [(operation, (0, 1))]))
        last[a] = last[b] = len(ops) - 1
    return ops


# -- routing ----------------------------------------------------------------------------------


@dataclass(slots=True)
class PhysOp:
    """An op of the routed circuit on grid positions.

    Attributes:
        kind: ``"one"``, ``"two"``, ``"exchange"`` or ``"swap"``.
        positions: Grid positions acted on.
        source: The logical op (``None`` for routing SWAPs).
        logical: Logical qubits involved (empty for SWAPs).
        merged_swap: Exchange block that also performs a SWAP of its two positions.
    """

    kind: Literal["one", "two", "exchange", "swap"]
    positions: tuple[Position, ...]
    source: LogicalOp | None = None
    logical: tuple[int, ...] = ()
    merged_swap: bool = False


Swap = tuple[Position, Position]


@dataclass(frozen=True, slots=True)
class MoveRules:
    """What a strategy contributes to routing.

    Attributes:
        name: Strategy name.
        candidates: ``candidates(u_home, v_home)`` returns SWAPs (pairs of grid-adjacent
            positions, in *home* coordinates) that may make the two qubits neighbours, best
            first.
        auxiliary_ok: ``auxiliary_ok(position)`` says whether an empty position may host an
            auxiliary qubit.
    """

    name: str
    candidates: Callable[[Position, Position], list[Swap]]
    auxiliary_ok: Callable[[Position], bool]


def _adjacent(p: Position, q: Position) -> bool:
    return abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1


def _norm(swap: Swap) -> Swap:
    a, b = swap
    return (a, b) if a <= b else (b, a)


@dataclass(slots=True)
class RoutedCircuit:
    """Output of :class:`SwapRouter`.

    Attributes:
        ops: Routed ops in program order.
        num_logical: Number of logical qubits.
        home: Initial grid position of each logical qubit.
        final: Grid position of each logical qubit at the end of the circuit.
        auxiliary: Grid positions used that host no logical qubit at the start.
        swaps: Routing SWAPs emitted (after elision).
        elided_swaps: SWAPs dropped because both qubits were still untouched ``|0>``.
        deferred_restores: SWAPs undone because a later block needed the original arrangement.
    """

    ops: list[PhysOp]
    num_logical: int
    home: tuple[Position, ...]
    final: tuple[Position, ...]
    auxiliary: tuple[Position, ...]
    swaps: int
    elided_swaps: int
    deferred_restores: int

    @property
    def positions(self) -> tuple[Position, ...]:
        """All grid positions used, sorted; index in this tuple = circuit qubit."""
        used = set(self.home) | set(self.auxiliary)
        for op in self.ops:
            used.update(op.positions)
        return tuple(sorted(used))

    def index(self) -> dict[Position, int]:
        """Grid position -> circuit qubit."""
        return {p: i for i, p in enumerate(self.positions)}

    def used_edges(self) -> set[tuple[int, int]]:
        """Circuit-qubit pairs touched by a two-qubit op (the coupling the hardware must offer)."""
        idx = self.index()
        edges: set[tuple[int, int]] = set()
        for op in self.ops:
            if len(op.positions) == 2:
                a, b = idx[op.positions[0]], idx[op.positions[1]]
                edges.add((min(a, b), max(a, b)))
        return edges

    def final_qubits(self) -> tuple[int, ...]:
        """Circuit qubit hosting each logical qubit at the end."""
        idx = self.index()
        return tuple(idx[p] for p in self.final)

    def auxiliary_qubits(self) -> tuple[int, ...]:
        """Circuit qubits that host no logical qubit at the end (they are in ``|0>``)."""
        hosts = set(self.final_qubits())
        return tuple(i for i in range(len(self.positions)) if i not in hosts)


class _RouterState:
    """Mutable bookkeeping of one routing pass (positions, active SWAPs, emitted ops)."""

    def __init__(self, home: tuple[Position, ...], rules: MoveRules) -> None:
        self.n = len(home)
        self.rules = rules
        # content[pos] = token at pos; tokens 0..n-1 are logical qubits, others auxiliary
        self.content: dict[Position, int] = {p: q for q, p in enumerate(home)}
        self.where: dict[int, Position] = dict(enumerate(home))
        self.home_of: dict[int, Position] = dict(self.where)
        self.active: list[Swap] = []
        self.touched: set[int] = set()
        self.aux: dict[Position, int] = {}
        self.out: list[PhysOp] = []
        self.swaps = 0
        self.elided = 0
        self.restores = 0

    def token_at(self, pos: Position) -> int:
        """Token at ``pos``, creating an auxiliary qubit there on first use."""
        if pos not in self.content:
            if not self.rules.auxiliary_ok(pos):
                msg = f"{self.rules.name}: position {pos} may not host an auxiliary qubit"
                raise MappingError(msg)
            token = self.n + len(self.aux)
            self.aux[pos] = token
            self.content[pos] = token
            self.where[token] = pos
            self.home_of[token] = pos
        return self.content[pos]

    def toggle(self, swap: Swap, *, restore: bool) -> None:
        """Exchange the contents of two positions; emit a SWAP unless both are still |0>."""
        a, b = swap
        ta, tb = self.token_at(a), self.token_at(b)
        self.content[a], self.content[b] = tb, ta
        self.where[ta], self.where[tb] = b, a
        if restore:
            self.active.remove(next(s for s in self.active if _norm(s) == swap))
        else:
            self.active.append(swap)
        if ta in self.touched or tb in self.touched:
            self.out.append(PhysOp("swap", (a, b)))
            self.swaps += 1
            self.restores += int(restore)
        else:
            self.elided += 1

    def displaced(self, token: int) -> list[Swap]:
        """Active SWAPs that moved ``token`` away from its home."""
        home = self.home_of[token]
        return [s for s in self.active if home in s]

    def moved(self, undo: list[Swap], move: Swap, tokens: tuple[int, int]) -> bool:
        """Whether undoing ``undo`` and then applying ``move`` makes ``tokens`` neighbours."""
        pos = {t: self.where[t] for t in tokens}
        for a, b in [*undo, move]:
            for t in tokens:
                if pos[t] == a:
                    pos[t] = b
                elif pos[t] == b:
                    pos[t] = a
        return _adjacent(pos[tokens[0]], pos[tokens[1]])

    def plan_for(self, u: int, v: int) -> list[tuple[Swap, bool]] | None:
        """Cheapest toggles ``(swap, is_restore)`` making ``u``, ``v`` neighbours, or ``None``."""
        if _adjacent(self.where[u], self.where[v]):
            return []
        options: list[list[tuple[Swap, bool]]] = []
        undo = {_norm(s) for s in self.displaced(u) + self.displaced(v)}
        if _adjacent(self.home_of[u], self.home_of[v]):
            options.append([(s, True) for s in sorted(undo)])
        for candidate in self.rules.candidates(self.home_of[u], self.home_of[v]):
            move = _norm(candidate)
            if not _adjacent(*move) or move in self.active:
                continue
            conflicts = sorted({_norm(s) for s in self.active if set(s) & set(move)} | undo)
            if self.moved(conflicts, move, (u, v)):
                options.append([(s, True) for s in conflicts] + [(move, False)])
        return min(options, key=len) if options else None

    def apply(self, op: LogicalOp) -> None:
        """Route one logical op: make its qubits neighbours, then emit it."""
        if op.kind == "one":
            (q,) = op.qubits
            self.out.append(PhysOp("one", (self.where[q],), op, op.qubits))
            self.touched.add(q)
            return
        u, v = op.qubits
        plan = self.plan_for(u, v)
        if plan is None:
            msg = (
                f"{self.rules.name}: no SWAP sequence makes logical qubits {u} (home "
                f"{self.home_of[u]}) and {v} (home {self.home_of[v]}) neighbours; the bond is "
                "outside what this embedding supports"
            )
            raise MappingError(msg)
        for swap, is_restore in plan:
            self.toggle(swap, restore=is_restore)
        self.out.append(PhysOp(op.kind, (self.where[u], self.where[v]), op, (u, v)))
        self.touched.update((u, v))


class SwapRouter:
    """Route logical blocks on a square grid with strategy-specific SWAP candidates.

    Args:
        home: Grid position of each logical qubit at the start.
        rules: The strategy's :class:`MoveRules`.
        restore_at_end: Undo all outstanding SWAPs at the end so that the final layout equals
            the initial one (costs a final SWAP layer).
    """

    def __init__(
        self, home: Sequence[Position], rules: MoveRules, *, restore_at_end: bool = False
    ) -> None:
        """See class docstring."""
        if len(set(home)) != len(home):
            msg = "two logical qubits share a home position"
            raise MappingError(msg)
        self._home = tuple(home)
        self._rules = rules
        self._restore = restore_at_end

    def route(self, ops: Sequence[LogicalOp]) -> RoutedCircuit:
        """Route ``ops`` (from :func:`logical_ops`) in their original order."""
        state = _RouterState(self._home, self._rules)
        for op in ops:
            state.apply(op)
        if self._restore:
            for swap in list(reversed(state.active)):
                state.toggle(_norm(swap), restore=True)
        return RoutedCircuit(
            ops=state.out,
            num_logical=state.n,
            home=self._home,
            final=tuple(state.where[q] for q in range(state.n)),
            auxiliary=tuple(sorted(state.aux)),
            swaps=state.swaps,
            elided_swaps=state.elided,
            deferred_restores=state.restores,
        )


# -- synthesis --------------------------------------------------------------------------------


def exchange_cx3(
    circuit: QuantumCircuit, ax: Angle, ay: Angle, az: Angle, q0: int, q1: int
) -> None:
    """Append ``exp(-i/2 (ax XX + ay YY + az ZZ))`` with three CNOTs (up to global phase).

    Vatan & Williams, PRA 69, 032315 (2004), Fig. 6, with signs fixed for Qiskit's conventions.
    The angles may be parameter expressions, so the circuit stays parameterised.
    """
    half_pi = np.pi / 2
    circuit.rz(half_pi, q1)
    circuit.cx(q1, q0)
    circuit.rz(az - half_pi, q0)
    circuit.ry(ax - half_pi, q1)
    circuit.cx(q0, q1)
    circuit.ry(half_pi - ay, q1)
    circuit.cx(q1, q0)
    circuit.rz(-half_pi, q0)


def _merge_swaps(ops: list[PhysOp]) -> tuple[list[PhysOp], int]:
    """Fold SWAPs into a neighbouring exchange block on the same pair."""
    result = list(ops)
    merged = 0
    i = 0
    while i < len(result):
        op = result[i]
        if op.kind != "swap":
            i += 1
            continue
        pair = set(op.positions)
        partner = None
        for j in range(i - 1, -1, -1):  # most recent op on these positions
            if set(result[j].positions) & pair:
                partner = j
                break
        if partner is None or not _mergeable(result[partner], pair):
            partner = None
            for j in range(i + 1, len(result)):  # or the next one (SWAP commutes with G)
                if set(result[j].positions) & pair:
                    partner = j
                    break
            if partner is not None and not _mergeable(result[partner], pair):
                partner = None
        if partner is None:
            i += 1
            continue
        result[partner].merged_swap = True
        del result[i]
        merged += 1
    return result, merged


def _mergeable(op: PhysOp, pair: set[Position]) -> bool:
    return op.kind == "exchange" and set(op.positions) == pair and not op.merged_swap


def synthesise(
    routed: RoutedCircuit,
    *,
    exchange: Literal["cx3", "rotations"] = "cx3",
    merge_swaps: bool = True,
    name: str = "routed",
) -> tuple[QuantumCircuit, int]:
    """Build the routed circuit on ``len(routed.positions)`` qubits.

    Args:
        routed: Router output.
        exchange: ``"cx3"`` (3-CNOT synthesis) or ``"rotations"`` (``rzz ryy rxx``).
        merge_swaps: Fold SWAPs into adjacent exchange blocks on the same pair.
        name: Circuit name.

    Returns:
        The circuit and the number of merged SWAPs.
    """
    idx = routed.index()
    ops, merged = _merge_swaps(routed.ops) if merge_swaps else (list(routed.ops), 0)
    circuit = QuantumCircuit(len(idx), name=name)
    for op in ops:
        qubits = [idx[p] for p in op.positions]
        if op.kind == "swap":
            circuit.swap(*qubits)
            continue
        assert op.source is not None
        if op.kind == "exchange":
            ax, ay, az = op.source.angles
            if op.merged_swap:
                ax, ay, az = ax + np.pi / 2, ay + np.pi / 2, az + np.pi / 2
            if exchange == "cx3":
                exchange_cx3(circuit, ax, ay, az, *qubits)
            else:
                circuit.rzz(az, *qubits)
                circuit.ryy(ay, *qubits)
                circuit.rxx(ax, *qubits)
            continue
        for operation, local in op.source.instructions:
            circuit.append(operation, [qubits[k] for k in local])
    # merged SWAPs change no bookkeeping: G(theta) SWAP acts on the same two positions
    return circuit, merged


def routed_observable(observable: SparsePauliOp, routed: RoutedCircuit) -> SparsePauliOp:
    """``observable`` on logical qubits, moved to the circuit qubits that host them at the end."""
    return observable.apply_layout(list(routed.final_qubits()), len(routed.positions))


def verify_routed(
    logical: QuantumCircuit,
    routed_circuit: QuantumCircuit,
    routed: RoutedCircuit,
    values: Sequence[float],
    *,
    atol: float = 1e-8,
) -> float:
    """Fidelity between the logical state and the routed state, permutation undone.

    Binds ``values`` (in ``logical.parameters`` order) into both circuits, simulates, checks
    that every auxiliary qubit ends in ``|0>``, and returns ``|<psi_logical|psi_routed>|^2``
    with the routed state's logical qubits read from :meth:`RoutedCircuit.final_qubits`.

    Raises:
        AssertionError: If an auxiliary qubit is not in ``|0>``.
    """
    from qiskit.quantum_info import Statevector, partial_trace

    params = list(logical.parameters)
    binding = dict(zip(params, values, strict=True))
    psi = Statevector(logical.assign_parameters(binding))
    routed_bound = routed_circuit.assign_parameters(
        {p: binding[p] for p in routed_circuit.parameters}
    )
    phi = Statevector(routed_bound)
    aux = routed.auxiliary_qubits()
    if aux:
        rho_aux = partial_trace(phi, list(routed.final_qubits()))
        p0 = float(np.real(rho_aux.data[0, 0]))
        if abs(p0 - 1) > atol:
            msg = f"auxiliary qubits {aux} are not in |0> (population {p0:.3e})"
            raise AssertionError(msg)
    # reorder: logical qubit q lives on circuit qubit final[q]; project aux onto |0>
    final = routed.final_qubits()
    m = routed_circuit.num_qubits
    amps = phi.data.reshape([2] * m)  # axis k <-> qubit m-1-k (little endian)
    axes_logical = [m - 1 - final[q] for q in reversed(range(len(final)))]
    index: list[Any] = [slice(None)] * m
    for a in aux:
        index[m - 1 - a] = 0
    reduced = amps[tuple(index)]
    kept_axes = [ax for ax in range(m) if (m - 1 - ax) not in aux]
    order = [kept_axes.index(ax) for ax in axes_logical]
    vec = np.transpose(reduced, order).reshape(-1)
    return float(abs(np.vdot(psi.data, vec)) ** 2)
