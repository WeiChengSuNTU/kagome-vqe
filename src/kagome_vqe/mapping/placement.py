"""Place a routed grid circuit on real hardware connectivity.

A geometric strategy produces a circuit on abstract grid positions and the set of couplers it
needs (``MappingResult.coupling``). :func:`place_on_backend` finds hardware qubits for those
positions by *subgraph monomorphism* of the needed coupling graph into the backend's coupling
map (rustworkx VF2). Nothing assumes the device is a perfect grid: missing couplers or qubits
simply reduce the candidate embeddings, and a heavy-hex device typically has none, which is
reported as :class:`~kagome_vqe.mapping.core.MappingError`.

Among the candidates, the one with the lowest calibration cost is chosen: the sum over needed
couplers of (two-qubit error x number of two-qubit blocks on that coupler) plus the readout
error of every measured qubit. Missing calibration data counts as zero, and ties are broken by
the sorted physical-qubit tuple, so the choice is deterministic.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import rustworkx as rx

from kagome_vqe.mapping.core import MappingError

__all__ = ["Placement", "backend_coupling_graph", "place_on_backend"]

log = logging.getLogger(__name__)

DEFAULT_MAX_CANDIDATES = 2000
DEFAULT_CALL_LIMIT = 5_000_000


@dataclass(frozen=True, slots=True)
class Placement:
    """Hardware qubits for the circuit qubits of a routed mapping.

    Attributes:
        physical: ``physical[i]`` is the hardware qubit hosting circuit qubit ``i``.
        method: ``"explicit"`` (user layout, validated) or ``"vf2"``.
        score: Calibration cost of the chosen embedding (lower is better).
        candidates: Embeddings examined.
    """

    physical: tuple[int, ...]
    method: str
    score: float
    candidates: int

    def describe(self) -> dict[str, Any]:
        """JSON-friendly summary."""
        return {
            "physical_qubits": list(self.physical),
            "placement_method": self.method,
            "placement_score": self.score,
            "placement_candidates": self.candidates,
        }


def _two_qubit_ops(target: Any) -> list[str]:
    names = []
    for name in target.operation_names:
        try:
            if target.operation_from_name(name).num_qubits == 2:
                names.append(name)
        except (KeyError, AttributeError):
            continue
    return sorted(names)


def _edge_error(target: Any, ops: Sequence[str], a: int, b: int) -> float | None:
    """Lowest two-qubit error on ``a-b`` (either direction); ``None`` if no gate exists."""
    best: float | None = None
    for name in ops:
        props = target[name]
        for key in ((a, b), (b, a)):
            if key in props:
                inst = props[key]
                err = 0.0 if inst is None or inst.error is None else float(inst.error)
                best = err if best is None else min(best, err)
    return best


def _readout_error(target: Any, q: int) -> float:
    if "measure" not in target.operation_names:
        return 0.0
    inst = target["measure"].get((q,))
    return 0.0 if inst is None or inst.error is None else float(inst.error)


def backend_coupling_graph(backend: Any) -> tuple[rx.PyGraph, dict[tuple[int, int], float]]:
    """Undirected coupling graph of usable couplers and their two-qubit error.

    Couplers with error ``>= 1`` (IBM's marker for broken gates) are dropped.

    Raises:
        MappingError: If the backend exposes no coupling information.
    """
    target = getattr(backend, "target", None)
    if target is None:
        msg = f"backend {getattr(backend, 'name', backend)!r} has no Target"
        raise MappingError(msg)
    ops = _two_qubit_ops(target)
    if not ops:
        msg = f"backend {backend.name!r} has no two-qubit gates (unrestricted connectivity?)"
        raise MappingError(msg)
    graph: rx.PyGraph = rx.PyGraph(multigraph=False)
    graph.add_nodes_from(list(range(int(target.num_qubits))))
    errors: dict[tuple[int, int], float] = {}
    for name in ops:
        for qargs in target[name]:
            if qargs is None or len(qargs) != 2:
                continue
            a, b = sorted(int(q) for q in qargs)
            if (a, b) in errors:
                continue
            err = _edge_error(target, ops, a, b)
            if err is None or err >= 1:
                continue
            errors[(a, b)] = err
            graph.add_edge(a, b, err)
    return graph, errors


def place_on_backend(
    num_qubits: int,
    coupling: Sequence[tuple[int, int]],
    backend: Any,
    *,
    usage: Counter[tuple[int, int]] | None = None,
    measured: Sequence[int] = (),
    explicit: Sequence[int] | None = None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> Placement:
    """Choose hardware qubits for a routed circuit.

    Args:
        num_qubits: Circuit qubits to place.
        coupling: Circuit-qubit pairs that need a hardware coupler.
        backend: A ``BackendV2`` with a ``Target``.
        usage: Two-qubit blocks per coupler (weights the error score); defaults to 1 each.
        measured: Circuit qubits that are measured (their readout error enters the score).
        explicit: A user-specified layout to validate instead of searching.
        max_candidates: Stop after examining this many embeddings.

    Raises:
        MappingError: If no embedding exists or ``explicit`` violates the connectivity.
    """
    hardware, errors = backend_coupling_graph(backend)
    target = backend.target
    edges = sorted({(min(a, b), max(a, b)) for a, b in coupling})
    weights = usage or Counter(dict.fromkeys(edges, 1))

    def score(phys: Sequence[int]) -> float:
        total = 0.0
        for a, b in edges:
            key = (min(phys[a], phys[b]), max(phys[a], phys[b]))
            total += errors.get(key, 0.0) * weights.get((a, b), 1)
        total += sum(_readout_error(target, phys[q]) for q in measured)
        return total

    if explicit is not None:
        phys = tuple(int(q) for q in explicit)
        if len(phys) != num_qubits or len(set(phys)) != num_qubits:
            msg = f"physical_layout needs {num_qubits} distinct qubits, got {list(phys)}"
            raise MappingError(msg)
        missing = [
            (phys[a], phys[b])
            for a, b in edges
            if (min(phys[a], phys[b]), max(phys[a], phys[b])) not in errors
        ]
        if missing:
            msg = f"physical_layout lacks hardware couplers {missing[:6]} needed by the mapping"
            raise MappingError(msg)
        return Placement(phys, "explicit", score(phys), 1)

    pattern: rx.PyGraph = rx.PyGraph(multigraph=False)
    pattern.add_nodes_from(list(range(num_qubits)))
    pattern.add_edges_from_no_data(edges)
    best: tuple[float, tuple[int, ...]] | None = None
    examined = 0
    mappings = rx.vf2_mapping(
        hardware,
        pattern,
        subgraph=True,
        induced=False,
        id_order=True,
        call_limit=DEFAULT_CALL_LIMIT,
    )
    for mapping in mappings:
        inverse = {logical: physical for physical, logical in mapping.items()}
        phys = tuple(int(inverse[i]) for i in range(num_qubits))
        candidate = (score(phys), phys)
        if best is None or candidate < best:
            best = candidate
        examined += 1
        if examined >= max_candidates:
            break
    if best is None:
        msg = (
            f"no embedding of the {num_qubits}-qubit routed grid ({len(edges)} couplers) into "
            f"{backend.name}'s coupling map; the geometry does not fit this device"
        )
        raise MappingError(msg)
    log.info(
        "placed %d qubits on %s: score %.4f after %d candidates",
        num_qubits,
        backend.name,
        best[0],
        examined,
    )
    return Placement(best[1], "vf2", best[0], examined)
