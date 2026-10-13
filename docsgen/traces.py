"""Stepper traces for the deck (``slides/data/<id>.json``).

Trace kind ``snapshots`` (schema 1, an extension of the tn-slide-deck schema): every step
carries the *complete* visual state, so ``render(k)`` in ``slides/js/snapshot.js`` draws
``steps[k]`` and nothing else. That keeps backward clicks and PDF export trivially correct.

    {"schema": 1, "id": ..., "kind": "snapshots", "meta": {...},
     "nodes": {"0": [x, y], ...}, "edges": [["0", "1"], ...],
     "steps": [{"caption": str,
                "edges": {"a_b": cls},          # cls: accent | derived | side-b | ghost
                "extra": [["a", "b", cls]],     # non-lattice pairs, drawn dashed
                "nodes": {"id": cls},           # optional node classes
                "labels": {"id": str},          # optional node labels (default: id)
                "readouts": {"key": str}}]}

Energies shown in readouts are exact statevector expectation values computed here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from docsgen import geometry
from docsgen.paths import SLIDES_DATA
from kagome_vqe.ansatz import ring_bonds, singlet_pair
from kagome_vqe.hamiltonian import heisenberg_hamiltonian
from kagome_vqe.lattice import (
    KAGOME_HAMILTONIAN_CYCLE,
    KAGOME_UNIT_CELL_EDGES,
    Lattice,
    kagome_unit_cell,
)

__all__ = ["ALL", "partial_singlet_energy", "write_all"]

SCALE = 120.0


def _nodes() -> dict[str, list[float]]:
    # SVG y points down
    return {
        str(s): [round(x * SCALE, 2), round(-y * SCALE, 2)]
        for s, (x, y) in geometry.positions().items()
    }


def _edge_key(a: int, b: int) -> str:
    lo, hi = sorted((a, b))
    return f"{lo}_{hi}"


def _is_bond(a: int, b: int) -> bool:
    return frozenset((a, b)) in {frozenset(e) for e in KAGOME_UNIT_CELL_EDGES}


def partial_singlet_energy(lattice: Lattice, pairs: int) -> float:
    """Energy of singlets on the first ``pairs`` even ring bonds, other qubits in ``|0>``."""
    qc = QuantumCircuit(lattice.num_sites)
    _, even = ring_bonds(lattice.num_sites)
    for a, b in even[:pairs]:
        singlet_pair(qc, a, b)
    return float(np.real(Statevector(qc).expectation_value(heisenberg_hamiltonian(lattice))))


def _fmt(x: float) -> str:
    return f"{x:+.1f}".replace("-", "−") if abs(x) > 1e-12 else "0.0"


def ring_build() -> dict[str, Any]:
    """Why the circuit works: lay the ring along a Hamiltonian cycle, then drop in singlets."""
    lattice = kagome_unit_cell()
    order = KAGOME_HAMILTONIAN_CYCLE
    n = len(order)
    qubit_label = {str(s): f"q{k}" for k, s in enumerate(order)}
    ring = [(order[k], order[(k + 1) % n]) for k in range(n)]
    odd, even = ring_bonds(n)
    e_up = _fmt(partial_singlet_energy(lattice, 0))  # |0...0>: every bond has ZZ = +1
    steps: list[dict[str, Any]] = [
        {
            "caption": "The 12-site kagome star: 18 bonds, 6 corner-sharing triangles.",
            "edges": {},
            "readouts": {"ring": "0 / 12", "singlets": "0 / 6", "energy": e_up},
        }
    ]
    edges: dict[str, str] = {}
    for k, (a, b) in enumerate(ring, start=1):
        edges = edges | {_edge_key(a, b): "derived"}
        steps.append(
            {
                "caption": f"Ring bond q{k - 1}–q{k % n}: sites {a}–{b} share a lattice bond.",
                "edges": dict(edges),
                "labels": qubit_label,
                "readouts": {"ring": f"{k} / 12", "singlets": "0 / 6", "energy": e_up},
            }
        )
    for j, (qa, qb) in enumerate(even, start=1):
        a, b = order[qa], order[qb]
        edges = edges | {_edge_key(a, b): "accent"}
        energy = partial_singlet_energy(lattice, j)
        steps.append(
            {
                "caption": f"Singlet on q{qa}, q{qb} (sites {a}–{b}): one per triangle.",
                "edges": dict(edges),
                "labels": qubit_label,
                "readouts": {"ring": "12 / 12", "singlets": f"{j} / 6", "energy": _fmt(energy)},
            }
        )
    odd_edges = {_edge_key(order[qa], order[qb]): "side-b" for qa, qb in odd}
    final = edges | odd_edges
    steps.append(
        {
            "caption": (
                "The odd ring bonds are the other dimer covering: the ring is the union of the two."
            ),
            "edges": final,
            "labels": qubit_label,
            "readouts": {
                "ring": "12 / 12",
                "singlets": "6 / 6",
                "energy": _fmt(partial_singlet_energy(lattice, 6)),
            },
        }
    )
    return {
        "schema": 1,
        "id": "ring_build",
        "kind": "snapshots",
        "meta": {"title": "Ring ansatz laid along a Hamiltonian cycle", "site_order": list(order)},
        "nodes": _nodes(),
        "edges": [[str(a), str(b)] for a, b in KAGOME_UNIT_CELL_EDGES],
        "steps": steps,
    }


def site_order() -> dict[str, Any]:
    """Identity order versus Hamiltonian cycle: which ring pairs are real bonds."""
    steps = []
    for name, order in (
        ("identity", tuple(range(12))),
        ("Hamiltonian cycle", KAGOME_HAMILTONIAN_CYCLE),
    ):
        lattice = kagome_unit_cell(site_order=None if name == "identity" else order)
        odd, even = ring_bonds(12)
        edges: dict[str, str] = {}
        extra: list[list[str]] = []
        for qa, qb in (*odd, *even):
            a, b = order[qa], order[qb]
            singlet = (qa, qb) in even
            cls = "accent" if singlet else "derived"
            if _is_bond(a, b):
                edges[_edge_key(a, b)] = cls
            else:
                extra.append([str(a), str(b), cls])
        on_ring = sum(_is_bond(order[k], order[(k + 1) % 12]) for k in range(12))
        on_singlet = sum(_is_bond(order[a], order[b]) for a, b in even)
        steps.append(
            {
                "caption": f"site_order = {name}",
                "edges": edges,
                "extra": extra,
                "labels": {str(s): f"q{k}" for k, s in enumerate(order)},
                "readouts": {
                    "ring": f"{on_ring} / 12",
                    "singlets": f"{on_singlet} / 6",
                    "energy": _fmt(partial_singlet_energy(lattice, 6)),
                },
            }
        )
    return {
        "schema": 1,
        "id": "site_order",
        "kind": "snapshots",
        "meta": {"title": "Identity order versus Hamiltonian cycle"},
        "nodes": _nodes(),
        "edges": [[str(a), str(b)] for a, b in KAGOME_UNIT_CELL_EDGES],
        "steps": steps,
    }


def validate(trace: dict[str, Any]) -> None:
    """Schema checks for ``snapshots`` traces (raises ``ValueError``)."""
    for key in ("schema", "id", "kind", "nodes", "edges", "steps"):
        if key not in trace:
            msg = f"{trace.get('id')}: missing key {key!r}"
            raise ValueError(msg)
    nodes = set(trace["nodes"])
    lattice_edges = {_edge_key(int(a), int(b)) for a, b in trace["edges"]}
    allowed = {"accent", "derived", "side-b", "ghost"}
    for i, step in enumerate(trace["steps"]):
        for key, cls in step.get("edges", {}).items():
            if key not in lattice_edges:
                msg = f"{trace['id']} step {i}: {key} is not a lattice edge"
                raise ValueError(msg)
            if cls not in allowed:
                msg = f"{trace['id']} step {i}: unknown class {cls}"
                raise ValueError(msg)
        for a, b, cls in step.get("extra", []):
            if a not in nodes or b not in nodes or cls not in allowed:
                msg = f"{trace['id']} step {i}: bad extra pair {a}-{b} {cls}"
                raise ValueError(msg)
            if _edge_key(int(a), int(b)) in lattice_edges:
                msg = f"{trace['id']} step {i}: extra pair {a}-{b} is a lattice edge"
                raise ValueError(msg)
        for v in (*step.get("labels", {}), *step.get("nodes", {})):
            if v not in nodes:
                msg = f"{trace['id']} step {i}: unknown node {v}"
                raise ValueError(msg)


ALL = {"ring_build": ring_build, "site_order": site_order}


def write_all() -> list[Path]:
    """Compute, validate and write every trace."""
    SLIDES_DATA.mkdir(parents=True, exist_ok=True)
    out = []
    for name, make in ALL.items():
        trace = make()
        validate(trace)
        path = SLIDES_DATA / f"{name}.json"
        path.write_text(json.dumps(trace, indent=1, ensure_ascii=False) + "\n")
        out.append(path)
    return out
