"""Compile-only comparison of hardware-mapping strategies (no circuit is executed).

:func:`mapping_benchmark` takes one run configuration and compiles the *same* logical circuit
with each requested strategy for the configured backend, then tabulates the resources. It
never submits a job: for ``backend.kind = "ibm"`` only the device's ``Target`` is fetched.

Metrics are reported at two levels:

* gate-block level (``block_2q_depth``, ``routing_swaps``): exchange gates and SWAPs counted as
  single two-qubit operations, the convention of the papers;
* native level (``native_2q_gates``, ``native_2q_depth``, ``depth``): after transpilation to
  the backend's basis, which is what the hardware runs.

For ``generic`` the SWAP count is the number SABRE inserts in an optimisation-level-1 routing
pass with the same seed and layout; the native metrics come from the configured level, where
the optimiser may absorb some of those SWAPs. Comparing strategies on one lattice and device is
meaningful; quoting one strategy as universally better is not.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kagome_vqe.ansatz import GraphHVAnsatz
from kagome_vqe.backends import resolve_backend
from kagome_vqe.config import RunConfig, Settings
from kagome_vqe.hamiltonian import heisenberg_hamiltonian
from kagome_vqe.mapping import MappingError
from kagome_vqe.transpile import IsaProblem

__all__ = ["MAPPING_COLUMNS", "mapping_benchmark", "write_mapping_benchmark"]

log = logging.getLogger(__name__)

MAPPING_COLUMNS: tuple[str, ...] = (
    "strategy",
    "status",
    "logical_qubits",
    "physical_qubits",
    "auxiliary_qubits",
    "cycles",
    "parameters",
    "exchange_gates",
    "routing_swaps",
    "merged_swaps",
    "block_2q_depth",
    "native_2q_gates",
    "native_2q_depth",
    "depth",
)


def _active_qubits(problem: IsaProblem) -> int:
    circuit = problem.circuit
    return len({circuit.find_bit(q).index for inst in circuit.data for q in inst.qubits})


def _sabre_swaps(problem_circuit: Any, backend: Any, seed: int, layout: Any) -> int | None:
    """SWAPs SABRE inserts for ``problem_circuit`` (level-1 routing, swaps kept as gates)."""
    if backend is None or getattr(backend, "coupling_map", None) is None:
        return None
    from qiskit.transpiler import generate_preset_pass_manager

    pm = generate_preset_pass_manager(
        optimization_level=1,
        coupling_map=backend.coupling_map,
        basis_gates=["cx", "rz", "ry", "rx", "h", "x", "sx", "swap"],
        initial_layout=layout,
        seed_transpiler=seed,
    )
    routed = pm.run(problem_circuit)
    return int(routed.count_ops().get("swap", 0))


def mapping_benchmark(
    cfg: RunConfig,
    strategies: Sequence[str] = ("prb105", "prb106_grid", "generic"),
    *,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Compile ``cfg``'s logical circuit with each strategy and collect resource counts.

    A strategy that does not apply is reported with ``status = "unsupported"`` and the reason,
    unless ``cfg.mapping.fallback = "generic"``, in which case the fallback is compiled and
    labelled ``generic_fallback``.
    """
    from kagome_vqe.mapping import map_circuit
    from kagome_vqe.vqe import compile_problem

    lattice = cfg.lattice.build()
    hamiltonian = heisenberg_hamiltonian(lattice)
    ansatz = cfg.ansatz.build(lattice)
    circuit = ansatz.build()
    backend = resolve_backend(cfg.backend, settings)
    rows: list[dict[str, Any]] = []
    for strategy in strategies:
        run_cfg = cfg.model_copy(
            update={"mapping": cfg.mapping.model_copy(update={"strategy": strategy})}
        )
        base: dict[str, Any] = {
            "strategy": strategy,
            "lattice": lattice.name,
            "boundary": lattice.boundary,
            "ansatz": ansatz.name,
            "schedule": ansatz.schedule.kind if isinstance(ansatz, GraphHVAnsatz) else None,
            "include_h0": ansatz.include_h0 if isinstance(ansatz, GraphHVAnsatz) else None,
            "backend": cfg.backend.label(),
            "optimization_level": cfg.backend.optimization_level,
            "logical_qubits": lattice.num_sites,
            "cycles": cfg.ansatz.reps,
            "parameters": ansatz.num_parameters,
        }
        start = time.perf_counter()
        try:
            problem = compile_problem(run_cfg, lattice, ansatz, circuit, hamiltonian, backend)
        except MappingError as exc:
            rows.append({**base, "status": "unsupported", "reason": str(exc)})
            log.info("%s unsupported: %s", strategy, exc)
            continue
        mapping = problem.mapping or {}
        used = mapping.get("used", "generic")
        routing_swaps = mapping.get("routing_swaps")
        if used == "generic" and backend is not None:
            generic = map_circuit("generic", lattice, circuit, hamiltonian)
            routing_swaps = _sabre_swaps(
                generic.circuit, backend, cfg.seed, cfg.backend.initial_layout
            )
        physical = mapping.get("physical_qubits")
        rows.append(
            {
                **base,
                "status": mapping.get("status", "generic"),
                "used": used,
                "physical_qubits": _active_qubits(problem)
                if backend is not None
                else mapping.get("circuit_qubits", lattice.num_sites),
                "auxiliary_qubits": mapping.get("auxiliary_qubits", 0),
                "exchange_gates": mapping.get("exchange_blocks"),
                "routing_swaps": routing_swaps,
                "merged_swaps": mapping.get("merged_swaps"),
                "block_2q_depth": mapping.get("block_2q_depth"),
                "native_2q_gates": problem.two_qubit_gates,
                "native_2q_depth": problem.two_qubit_depth,
                "depth": problem.depth,
                "gate_counts": problem.gate_counts,
                "placement": physical,
                "measured_on": None
                if problem.physical_qubits is None
                else list(problem.physical_qubits),
                "notes": mapping.get("notes", []),
                "compile_time_s": time.perf_counter() - start,
            }
        )
    return rows


def write_mapping_benchmark(rows: list[dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    """Write ``mapping.json`` and a Markdown table ``mapping.md`` into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "mapping.json"
    json_path.write_text(json.dumps(rows, indent=2, default=str) + "\n")
    lines = [
        "| " + " | ".join(MAPPING_COLUMNS) + " |",
        "|" + "---|" * len(MAPPING_COLUMNS),
    ]
    for row in rows:
        cells = ["-" if row.get(col) is None else str(row.get(col)) for col in MAPPING_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    notes = [
        f"- {row['strategy']}: {row.get('reason') or '; '.join(row.get('notes', []))}"
        for row in rows
        if row.get("reason") or row.get("notes")
    ]
    if rows:
        head = rows[0]
        lines.insert(
            0,
            f"Lattice `{head['lattice']}` ({head['boundary']}), ansatz `{head['ansatz']}` "
            f"(schedule `{head['schedule']}`, H0 {head['include_h0']}), "
            f"backend `{head['backend']}`, optimisation level {head['optimization_level']}.\n",
        )
    if notes:
        lines += ["", "Notes:", *notes]
    md_path = out_dir / "mapping.md"
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path
