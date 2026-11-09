"""The three hardware-mapping strategies and their common result type.

``prb105``
    Bosse & Montanaro, PRB 105, 094409 (2022), Figs. 2-3. Every square-grid qubit hosts a
    kagome site (no auxiliary qubits). Grid row ``g`` holds the horizontal kagome line
    ``r = 2g`` with the apex sites of row ``r - 1`` inserted between their two lower
    neighbours (:func:`~kagome_vqe.kagome_lattices.bm_coord_to_grid`). Two thirds of the
    kagome bonds are then grid edges. The remaining ones (their groups ``H4`` and ``H5``) become
    grid edges after swapping each apex site with its right-hand neighbour, which is the SWAP
    layer of their Fig. 3.
``prb106_grid``
    Kattemolle & van Wezel, PRB 106, 214429 (2022), Fig. 2 ("KVQE_G"). The sheared triangular
    lattice is laid on the grid as is; the hexagon centres (odd row, odd column) are
    *swapping stations*: auxiliary qubits in ``|0>``. A diagonal kagome bond is made a grid
    edge by swapping its lower-right endpoint into the adjacent station.
``generic``
    No geometric embedding: exchange blocks are synthesised with three CNOTs and Qiskit's
    preset pass manager does layout and routing (SABRE).

Both geometric strategies share :class:`~kagome_vqe.mapping.core.SwapRouter`, which walks the
logical gates in their original order, so any schedule is implemented correctly. A result is
labelled

* ``paper_exact`` when lattice *and* schedule are those of the paper, so the routed circuit is
  the paper's construction (the tests check the per-cycle depth the paper quotes);
* ``paper_inspired`` for the same embedding on another open patch or with another schedule;
  the paper's depth is then *not* implied;
* ``generic`` / ``generic_fallback`` for Qiskit routing (the latter when a geometric strategy
  was requested, did not apply, and fallback was explicitly allowed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from kagome_vqe.kagome_lattices import KVQE20_COORDS, bm_coord_to_grid, bm_grid_to_coord
from kagome_vqe.lattice import Lattice, is_kagome_hole
from kagome_vqe.mapping.core import (
    MappingError,
    MoveRules,
    PhysOp,
    Position,
    RoutedCircuit,
    SwapRouter,
    exchange_cx3,
    logical_ops,
    routed_observable,
    synthesise,
)
from kagome_vqe.schedule import ExchangeSchedule

__all__ = [
    "MAPPING_STRATEGIES",
    "MappingResult",
    "MappingStatus",
    "MappingStrategyName",
    "block_two_qubit_depth",
    "map_circuit",
    "presynthesise_exchange",
]

log = logging.getLogger(__name__)

MappingStrategyName = Literal["prb105", "prb106_grid", "generic"]
MappingStatus = Literal["paper_exact", "paper_inspired", "generic", "generic_fallback"]
MAPPING_STRATEGIES: tuple[str, ...] = ("prb105", "prb106_grid", "generic")


@dataclass(slots=True)
class MappingResult:
    """A logical circuit implemented on an abstract device graph (or left to Qiskit).

    Attributes:
        requested: Strategy asked for.
        used: Strategy actually applied.
        status: See the module docstring.
        circuit: For geometric strategies: the routed circuit on ``len(grid_positions)``
            qubits. For ``generic``: the logical circuit with exchange blocks synthesised.
        observable: The Hamiltonian on ``circuit``'s qubits, already permuted to the final
            placement for routed circuits.
        initial_qubits: Circuit qubit hosting each logical qubit at the start.
        final_qubits: Circuit qubit hosting each logical qubit at the end.
        auxiliary_qubits: Circuit qubits hosting no logical qubit at the end (``|0>``).
        grid_positions: Grid coordinate of each circuit qubit (``None`` for generic).
        coupling: Circuit-qubit pairs the circuit needs as hardware couplers.
        routed: Router output (``None`` for generic).
        stats: Resource counts at the gate-block level (before hardware transpilation).
        notes: Human-readable caveats (e.g. why a fallback happened).
    """

    requested: str
    used: str
    status: MappingStatus
    circuit: QuantumCircuit
    observable: SparsePauliOp
    initial_qubits: tuple[int, ...]
    final_qubits: tuple[int, ...]
    auxiliary_qubits: tuple[int, ...] = ()
    grid_positions: tuple[Position, ...] | None = None
    coupling: frozenset[tuple[int, int]] = frozenset()
    routed: RoutedCircuit | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def is_routed(self) -> bool:
        """Whether the circuit is already routed on a grid (no Qiskit routing needed)."""
        return self.routed is not None

    def describe(self) -> dict[str, Any]:
        """JSON-friendly summary."""
        return {
            "requested": self.requested,
            "used": self.used,
            "status": self.status,
            "circuit_qubits": self.circuit.num_qubits,
            "auxiliary_qubits": len(self.auxiliary_qubits),
            "grid_positions": None
            if self.grid_positions is None
            else [list(p) for p in self.grid_positions],
            "initial_qubits": list(self.initial_qubits),
            "final_qubits": list(self.final_qubits),
            **self.stats,
            "notes": list(self.notes),
        }


# -- helpers ----------------------------------------------------------------------------------


def presynthesise_exchange(circuit: QuantumCircuit) -> QuantumCircuit:
    """Replace each ``rzz ryy rxx`` exchange block by its 3-CNOT synthesis, nothing else."""
    out = QuantumCircuit(circuit.num_qubits, name=circuit.name)
    for op in logical_ops(circuit):
        if op.kind == "exchange":
            exchange_cx3(out, *op.angles, *op.qubits)
            continue
        for operation, local in op.instructions:
            out.append(operation, [op.qubits[k] for k in local])
    return out


def block_two_qubit_depth(ops: list[PhysOp]) -> int:
    """Two-qubit depth counting each exchange block, SWAP or other two-qubit op as one layer.

    This is the metric the papers quote ("assuming the Heisenberg gate and SWAP are native").
    A SWAP merged into an exchange block counts once.
    """
    level: dict[Position, int] = {}
    depth = 0
    for op in ops:
        if len(op.positions) != 2:
            continue
        a, b = op.positions
        layer = max(level.get(a, 0), level.get(b, 0)) + 1
        level[a] = level[b] = layer
        depth = max(depth, layer)
    return depth


def _count(ops: list[PhysOp], kind: str) -> int:
    return sum(op.kind == kind for op in ops)


def _translate_even(positions: list[Position]) -> list[Position]:
    """Shift so the minimum row/column is 0 or 1, keeping both parities (holes stay holes)."""
    r0 = min(p[0] for p in positions)
    c0 = min(p[1] for p in positions)
    r0 -= r0 % 2
    c0 -= c0 % 2
    return [(r - r0, c - c0) for r, c in positions]


def _require_geometry(lattice: Lattice, strategy: str) -> tuple[tuple[int, int], ...]:
    if lattice.boundary == "periodic":
        msg = (
            f"{strategy} embeds open patches only; {lattice.name} is periodic (wrapped bonds are"
            " not grid-local)"
        )
        raise MappingError(msg)
    if lattice.coords is None:
        msg = f"{strategy} needs triangular-lattice coordinates; {lattice.name} has none"
        raise MappingError(msg)
    if not lattice.is_kagome_geometry():
        msg = f"{strategy}: some bonds of {lattice.name} are not kagome nearest neighbours"
        raise MappingError(msg)
    return lattice.coords


def _logical_positions(lattice: Lattice, site_positions: list[Position]) -> list[Position]:
    """Home position per ansatz qubit (``site_order`` applied)."""
    order = lattice.site_order or tuple(range(lattice.num_sites))
    return [site_positions[order[q]] for q in range(lattice.num_sites)]


# -- PRB 106 ----------------------------------------------------------------------------------


def _prb106_rules() -> MoveRules:
    def candidates(u: Position, v: Position) -> list[tuple[Position, Position]]:
        p, q = (u, v) if u <= v else (v, u)
        if (q[0] - p[0], q[1] - p[1]) != (1, 1):
            return []
        r, c = p
        stations = [s for s in ((r + 1, c), (r, c + 1)) if is_kagome_hole(s)]
        # canonical (paper): move the lower-right endpoint into the station
        return [(s, q) for s in stations] + [(s, p) for s in stations]

    return MoveRules("prb106_grid", candidates, is_kagome_hole)


def _is_kvqe20(coords: list[Position]) -> bool:
    return set(_translate_even(coords)) == set(KVQE20_COORDS)


# -- PRB 105 ----------------------------------------------------------------------------------


def _prb105_rules(apex: set[Position]) -> MoveRules:
    def candidates(u: Position, v: Position) -> list[tuple[Position, Position]]:
        # canonical (paper): swap an apex with its right-hand neighbour. H4 bonds contain the
        # apex itself; H5 bonds straddle it (x, apex, x + 2), and the apex swaps with x + 2.
        moves: list[tuple[Position, Position]] = [
            (x, (x[0], x[1] + 1)) for x in (u, v) if x in apex
        ]
        for x, y in ((u, v), (v, u)):
            mid = (x[0], x[1] + 1)
            if y == (x[0], x[1] + 2) and mid in apex and (mid, y) not in moves:
                moves.append((mid, y))
        for x in (u, v):
            for d in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                y = (x[0] + d[0], x[1] + d[1])
                if (x, y) not in moves:
                    moves.append((x, y))
        return moves

    return MoveRules("prb105", candidates, lambda _pos: True)


def _bm_rectangle_shape(grid: list[Position]) -> tuple[int, int] | None:
    """``(rows, cols)`` if ``grid`` is exactly a BM rectangle in canonical placement."""
    rows = max(g for g, _ in grid) - min(g for g, _ in grid) + 1
    cols = max(j for _, j in grid) - min(j for _, j in grid) + 1
    if len(grid) != rows * cols:
        return None
    g0, j0 = min(g for g, _ in grid), min(j for _, j in grid)
    coords = {bm_grid_to_coord(g, j) for g in range(g0, g0 + rows) for j in range(j0, j0 + cols)}
    if {bm_grid_to_coord(*p) for p in grid} != coords:
        return None
    return rows, cols


# -- dispatch ---------------------------------------------------------------------------------


def _geometric(
    strategy: Literal["prb105", "prb106_grid"],
    lattice: Lattice,
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    schedule: ExchangeSchedule | None,
    *,
    restore_at_end: bool,
    merge_swaps: bool,
    exchange: Literal["cx3", "rotations"],
) -> MappingResult:
    coords = list(_require_geometry(lattice, strategy))
    notes: list[str] = []
    if strategy == "prb106_grid":
        site_pos = _translate_even(coords)
        rules = _prb106_rules()
        exact = _is_kvqe20(coords) and schedule is not None and schedule.kind == "prb106"
        reference = "PRB 106 Fig. 2, 20-site patch with the paper's six-layer cycle"
    else:
        grid = [bm_coord_to_grid(c) for c in coords]
        g0, j0 = min(g for g, _ in grid), min(j for _, j in grid)
        site_pos = [(g - g0, j - j0) for g, j in grid]
        apex = {site_pos[i] for i, c in enumerate(coords) if c[0] % 2 == 1}
        rules = _prb105_rules(apex)
        shape = _bm_rectangle_shape(grid)
        exact = (
            shape is not None
            and shape[1] % 2 == 0
            and schedule is not None
            and schedule.kind in ("prb105", "prb105_fig3")
        )
        reference = (
            f"PRB 105 Figs. 2-3 on the {shape[0]}x{shape[1]} rectangle"
            if shape
            else "PRB 105 Figs. 2-3"
        )
    if not exact:
        notes.append(
            f"paper-inspired: the {strategy} embedding applied outside the paper's setting "
            f"({reference}); the paper's circuit depth is not implied"
        )
    router = SwapRouter(_logical_positions(lattice, site_pos), rules, restore_at_end=restore_at_end)
    routed = router.route(logical_ops(circuit))
    built, merged = synthesise(routed, exchange=exchange, merge_swaps=merge_swaps, name=strategy)
    idx = routed.index()
    stats = {
        "logical_qubits": routed.num_logical,
        "physical_qubits": len(routed.positions),
        "exchange_blocks": _count(routed.ops, "exchange"),
        "routing_swaps": routed.swaps,
        "merged_swaps": merged,
        "elided_swaps": routed.elided_swaps,
        "restoring_swaps": routed.deferred_restores,
        "block_2q_depth": block_two_qubit_depth(routed.ops),
        "restore_at_end": restore_at_end,
        "paper_reference": reference,
    }
    return MappingResult(
        requested=strategy,
        used=strategy,
        status="paper_exact" if exact else "paper_inspired",
        circuit=built,
        observable=routed_observable(observable, routed),
        initial_qubits=tuple(idx[p] for p in routed.home),
        final_qubits=routed.final_qubits(),
        auxiliary_qubits=routed.auxiliary_qubits(),
        grid_positions=routed.positions,
        coupling=frozenset(routed.used_edges()),
        routed=routed,
        stats=stats,
        notes=notes,
    )


def _generic(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    *,
    requested: str,
    exchange: Literal["cx3", "rotations"],
    notes: list[str],
) -> MappingResult:
    built = presynthesise_exchange(circuit) if exchange == "cx3" else circuit.copy()
    n = circuit.num_qubits
    ops = logical_ops(circuit)
    return MappingResult(
        requested=requested,
        used="generic",
        status="generic" if requested == "generic" else "generic_fallback",
        circuit=built,
        observable=observable,
        initial_qubits=tuple(range(n)),
        final_qubits=tuple(range(n)),
        stats={
            "logical_qubits": n,
            "physical_qubits": None,
            "exchange_blocks": sum(op.kind == "exchange" for op in ops),
            "routing_swaps": None,
            "block_2q_depth": None,
            "paper_reference": None,
        },
        notes=notes,
    )


def map_circuit(
    strategy: MappingStrategyName,
    lattice: Lattice,
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    *,
    schedule: ExchangeSchedule | None = None,
    fallback: Literal["error", "generic"] = "error",
    restore_at_end: bool = False,
    merge_swaps: bool = True,
    exchange: Literal["cx3", "rotations"] = "cx3",
) -> MappingResult:
    """Implement ``circuit`` (on ``lattice.num_sites`` logical qubits) with ``strategy``.

    Args:
        strategy: ``"prb105"``, ``"prb106_grid"`` or ``"generic"``.
        lattice: The interaction graph (geometry needed by the paper strategies).
        circuit: Logical circuit, e.g. ``GraphHVAnsatz.build()`` (possibly parameter-unshared).
        observable: Hamiltonian on the logical qubits.
        schedule: The ansatz's schedule; only used to decide ``paper_exact``.
        fallback: ``"error"`` (default) raises :class:`MappingError` when a geometric strategy
            does not apply; ``"generic"`` falls back to Qiskit routing and says so in the
            result's ``status`` and ``notes``.
        restore_at_end: Undo outstanding SWAPs at the end (identity final permutation).
        merge_swaps: Fold SWAPs into adjacent exchange blocks on the same pair.
        exchange: Exchange-block synthesis.

    Raises:
        MappingError: If the strategy does not apply and ``fallback == "error"``.
        ValueError: For an unknown strategy.
    """
    if circuit.num_qubits != lattice.num_sites:
        msg = f"circuit has {circuit.num_qubits} qubits, lattice {lattice.num_sites} sites"
        raise ValueError(msg)
    if strategy == "generic":
        return _generic(circuit, observable, requested="generic", exchange=exchange, notes=[])
    if strategy not in ("prb105", "prb106_grid"):
        msg = f"unknown mapping strategy {strategy!r}; choose from {MAPPING_STRATEGIES}"
        raise ValueError(msg)
    try:
        return _geometric(
            strategy,
            lattice,
            circuit,
            observable,
            schedule,
            restore_at_end=restore_at_end,
            merge_swaps=merge_swaps,
            exchange=exchange,
        )
    except MappingError as exc:
        if fallback != "generic":
            raise
        log.warning("%s not applicable (%s); falling back to generic routing", strategy, exc)
        return _generic(
            circuit,
            observable,
            requested=strategy,
            exchange=exchange,
            notes=[f"fallback to generic routing: {exc}"],
        )
