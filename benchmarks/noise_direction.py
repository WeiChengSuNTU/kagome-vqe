"""Does gate noise change the *direction* of the energy gradient? Exact, without shot noise.

Usage::

    uv run python benchmarks/noise_direction.py plan                  # points, models, cost
    uv run python benchmarks/noise_direction.py run MODEL POINT [POINT ...] [--threads 2]
    uv run python benchmarks/noise_direction.py report [--out results/<UTC>-noise_direction]

Question (test A of ``docs/experiments/graph_hva_pidn_next.md``): mitigation, and a surrogate
of a mitigated optimiser (PIDN of ZNE-GD), can only beat plain gradient descent if the noise
*changes where the gradient points*. On PRB 106 layer noise the bias was found to be nearly
parameter-independent. Is that still true for compiled hardware noise (gate errors, T1/T2
relaxation, readout) after routing?

For every point (noise-free L-BFGS-B optima, ``p = 1, 2, 3``, seeds 1-3, and three points of
one noisy gradient-descent trajectory) and every noise model, this script computes from exact
expectation values (Aer density matrices, ``save_expectation_value``; no shots):

* ``E_ideal`` and ``g_ideal`` (statevector, exchange shift rule);
* ``E_noisy`` and ``g_noisy``: gate noise only (= perfectly readout-corrected);
  ``g`` by central finite differences, ``h = 1e-3`` (truncation error ~1e-7);
* ``E_ro`` and ``g_ro``: the same plus the device's readout error, applied analytically per
  qubit (asymmetric flips: ``E[z'] = a + b z``), i.e. what an unmitigated estimator converges
  to with infinitely many shots;
* cosines with ``g_ideal``, norms, the best scalar ``c = g.g_ideal / |g_ideal|^2`` and the
  part of ``g`` that no rescaling of ``g_ideal`` explains, ``|g - c g_ideal|``.

Noise models (12 active qubits each, so that the density matrix is 268 MB):

``layer``
    PRB 106 paper layers, depolarising ``p_e = 1e-3`` on every qubit after the singlet
    preparation and after every exchange layer (the control: the model of all earlier work),
    symmetric readout ``q = 0.02``.
``nighthawk_prb105``
    FakeNighthawk (square lattice) with the PRB 105 square-grid embedding (``paper_exact`` for
    the 2x6 rectangle with the ``prb105`` schedule), placement by calibrated error.
``guadalupe_ring``
    FakeGuadalupeV2, the 12-qubit heavy-hex ring of main's M2 (``initial_layout``), SABRE
    routing restricted to the ring.
``kingston_ring``
    FakeKingston (Heron r2), the 12-qubit hexagon with the lowest mean CZ error, routing
    restricted to it. (Main's M2 used FakeGuadalupeV2 only; Kingston is the current device.)
``kingston_median``
    The same on the hexagon of median mean CZ error (rank 10 of 19).

Generic routing on the full coupling maps uses 13 (Guadalupe) and 14 (Kingston) qubits, whose
density matrices are 4x and 16x larger; the ring restriction keeps 12 at the price of more
SWAPs (190 CX / 190 CZ at ``p = 2`` instead of 144 / 147).

Results are cached per (model, point) under ``results/noise_direction_cache/`` in chunks, so
a process can be interrupted and resumed; ``report`` writes
``results/<UTC>-noise_direction/{points.csv, report.md}``.
"""

from __future__ import annotations

import argparse
import ast
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector

from kagome_vqe.config import RunConfig, apply_overrides, load_toml
from kagome_vqe.hamiltonian import heisenberg_hamiltonian
from kagome_vqe.noise import aer_circuit, layer_noise_circuit
from kagome_vqe.results import load_optimal_params

log = logging.getLogger("noise_direction")

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "noise_direction_cache"
H_FD = 1e-3
CHUNK = 8
LAYER_P = 1e-3
LAYER_Q = 0.02
MODELS = ("layer", "nighthawk_prb105", "guadalupe_ring", "kingston_ring", "kingston_median")
GUADALUPE_RING = [1, 2, 3, 5, 8, 11, 14, 13, 12, 10, 7, 4]  # main's M2 initial_layout
TRANSPILE_SEED = 1

_OPT = {
    1: "results/*-graph_hva_opt_repr/runs/*-graph_hva_opt_repr-bm2x6-p1-noh0-sv-lbfgsb-s{s}",
    2: "results/*-graph_hva_opt_repr/runs/*-graph_hva_opt_repr-bm2x6-p2-noh0-sv-lbfgsb-s{s}",
    3: "results/*-graph_hva_noise_reference/runs/*-bm2x6-p3-noh0-sv-lbfgsb-s{s}",
}
_GD = "results/*-graph_hva_m2_gd/runs/*-graph_hva_m2_gd-bm2x6-p2-noh0-full_rs-gd_ps60-s{s}"
GD_ITERATIONS = {1: (0, 10, 60), 2: (10,), 3: (10,)}


@dataclass(frozen=True, slots=True)
class Point:
    """A parameter vector to analyse."""

    label: str
    reps: int
    theta: np.ndarray
    source: str


def _one(pattern: str) -> Path:
    hits = sorted(ROOT.glob(pattern))
    if not hits:
        msg = f"no run matches {pattern}"
        raise FileNotFoundError(msg)
    return hits[-1]


def points() -> dict[str, Point]:
    """Optima of p = 1, 2, 3 (seeds 1-3); noisy GD (readout + symmetry, 1024 shots) iterates."""
    out: dict[str, Point] = {}
    for reps, pattern in _OPT.items():
        for seed in (1, 2, 3):
            run = _one(pattern.format(s=seed))
            out[f"opt_p{reps}_s{seed}"] = Point(
                f"opt_p{reps}_s{seed}", reps, load_optimal_params(run), str(run.relative_to(ROOT))
            )
    for seed, iterations in GD_ITERATIONS.items():
        run = _one(_GD.format(s=seed))
        iterates = np.load(run / "iterates.npy")
        for it in iterations:
            label = f"gd_p2_s{seed}_it{it}"
            out[label] = Point(label, 2, iterates[it], f"{run.relative_to(ROOT)}#iterate{it}")
    return out


def _base_config(reps: int) -> RunConfig:
    data = load_toml(_one(_OPT[2].format(s=1)) / "config.toml")
    data = apply_overrides(data, [f"ansatz.reps={reps}", f"seed={TRANSPILE_SEED}"])
    return RunConfig.model_validate(data)


# -- compiled problems ---------------------------------------------------------------------------


@dataclass(slots=True)
class Compiled:
    """A parametrised circuit with 12 active qubits, its Hamiltonian terms and readout."""

    circuit: QuantumCircuit  # parameters in ansatz order (circuit.parameters)
    terms: list[tuple[float, str, int, int]]  # (coefficient, letter, qubit a, qubit b)
    readout: dict[int, tuple[float, float]]  # qubit -> (p(1|0), p(0|1))
    noise_model: Any | None
    info: dict[str, Any]


def _terms(observable: SparsePauliOp) -> list[tuple[float, str, int, int]]:
    out = []
    for label, coeff in observable.to_list():
        support = [(i, ch) for i, ch in enumerate(reversed(label)) if ch != "I"]
        if len(support) != 2 or support[0][1] != support[1][1]:
            msg = f"unexpected Hamiltonian term {label}"
            raise ValueError(msg)
        out.append((float(np.real(coeff)), support[0][1], support[0][0], support[1][0]))
    return out


def _readout(noise_model: Any, qubits: list[int]) -> dict[int, tuple[float, float]]:
    errors = noise_model._local_readout_errors
    out = {}
    for q in qubits:
        probs = np.asarray(errors[(q,)].probabilities)
        out[q] = (float(probs[0][1]), float(probs[1][0]))
    return out


def _active(circuit: QuantumCircuit) -> list[int]:
    return sorted({circuit.find_bit(q).index for inst in circuit.data for q in inst.qubits})


def _canonical_ring(ring: list[int]) -> tuple[int, ...]:
    """Rotate a cycle to start at its smallest qubit, oriented towards the smaller neighbour."""
    i = ring.index(min(ring))
    rotated = ring[i:] + ring[:i]
    if rotated[-1] < rotated[1]:
        rotated = [rotated[0], *reversed(rotated[1:])]
    return tuple(rotated)


def _ring_kingston(backend: Any, rank: str = "best") -> list[int]:
    """A heavy-hex hexagon (12-qubit cycle) ranked by mean CZ error, deterministically.

    ``rank``: ``"best"`` (lowest error) or ``"median"``.
    ``rustworkx.cycle_basis`` returns *a* basis whose cycles (and their orientation) differ
    between processes, which once made different jobs compile onto different hexagons. Here the
    candidates are the length-12 cycles of the bases rooted at every qubit, canonicalised, and
    ties are broken by the qubit tuple.
    """
    import rustworkx as rx

    graph = backend.coupling_map.graph.to_undirected(multigraph=False)
    target = backend.target
    rings: set[tuple[int, ...]] = set()
    for root in graph.node_indices():
        for cycle in rx.cycle_basis(graph, root):
            if len(cycle) == 12:
                rings.add(_canonical_ring(list(cycle)))

    def cz_error(ring: tuple[int, ...]) -> float:
        errs = []
        for a, b in zip(ring, ring[1:] + ring[:1], strict=True):
            props = target["cz"].get((a, b)) or target["cz"].get((b, a))
            errs.append(props.error)
        return float(np.mean(errs))

    ranked = sorted(rings, key=lambda r: (cz_error(r), r))
    return list(ranked[0] if rank == "best" else ranked[len(ranked) // 2])


def circuit_fingerprint(circuit: QuantumCircuit) -> str:
    """Short hash of a circuit's gate sequence and qubits (detects a different compilation)."""
    import hashlib

    ops = [
        (i.operation.name, tuple(circuit.find_bit(q).index for q in i.qubits)) for i in circuit.data
    ]
    return hashlib.sha1(repr(ops).encode()).hexdigest()[:12]


def compile_model(model: str, reps: int) -> Compiled:
    """Build the circuit of ``model`` at ``reps`` (deterministic: fixed seeds)."""
    from qiskit.transpiler import CouplingMap
    from qiskit_aer.noise import NoiseModel

    from kagome_vqe.backends import resolve_fake_backend
    from kagome_vqe.mapping import map_circuit
    from kagome_vqe.vqe import prepare

    cfg = _base_config(reps)
    lattice = cfg.lattice.build()
    hamiltonian = heisenberg_hamiltonian(lattice)
    ansatz = cfg.ansatz.build(lattice)
    if model == "layer":
        circuit = aer_circuit(layer_noise_circuit(ansatz, "depolarizing", LAYER_P))
        return Compiled(
            circuit,
            _terms(hamiltonian),
            dict.fromkeys(range(lattice.num_sites), (LAYER_Q, LAYER_Q)),
            None,
            {"model": model, "qubits": list(range(lattice.num_sites))},
        )
    if model == "nighthawk_prb105":
        data = cfg.model_dump(mode="json")
        data = apply_overrides(
            data,
            [
                'backend.kind="aer_noisy"',
                'backend.name="FakeNighthawk"',
                'mapping.strategy="prb105"',
            ],
        )
        prep = prepare(RunConfig.model_validate(data))
        isa, observable, backend = prep.problem.circuit, prep.problem.observable, prep.backend
        status = (prep.problem.mapping or {}).get("status")
    else:
        name = "FakeGuadalupeV2" if model == "guadalupe_ring" else "FakeKingston"
        backend = resolve_fake_backend(name)
        if model == "guadalupe_ring":
            ring = GUADALUPE_RING
        else:
            ring = _ring_kingston(backend, "median" if model == "kingston_median" else "best")
        edges = [(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]
        coupling = CouplingMap(edges + [(b, a) for a, b in edges])
        routed = map_circuit("generic", lattice, ansatz.build(), hamiltonian)
        isa = transpile(
            routed.circuit,
            backend=backend,
            coupling_map=coupling,
            initial_layout=ring,
            optimization_level=3,
            seed_transpiler=TRANSPILE_SEED,
        )
        observable = hamiltonian.apply_layout(isa.layout)
        status = f"ring {ring}"
    active = _active(isa)
    if len(active) != lattice.num_sites:
        msg = f"{model} p={reps}: {len(active)} active qubits, expected {lattice.num_sites}"
        raise RuntimeError(msg)
    noise_model = NoiseModel.from_backend(backend)
    counts = isa.count_ops()
    two_q = sum(v for k, v in counts.items() if k in ("cx", "cz", "ecr"))
    local = {q: i for i, q in enumerate(active)}
    readout = _readout(noise_model, active)
    return Compiled(
        _compact_circuit(isa, local),
        [(c, letter, local[a], local[b]) for c, letter, a, b in _terms(observable)],
        {local[q]: v for q, v in readout.items()},
        _compact_noise(noise_model, local),
        {
            "model": model,
            "device": str(backend.name),
            "mapping": status,
            "qubits": list(range(len(active))),
            "physical_qubits": active,
            "two_qubit_gates": int(two_q),
            "depth": int(isa.depth()),
        },
    )


def _compact_circuit(circuit: QuantumCircuit, local: dict[int, int]) -> QuantumCircuit:
    """The circuit on its active qubits only (physical qubit ``q`` -> ``local[q]``).

    Aer would truncate idle qubits itself, but its ``save_expectation_value`` on a *subset* of
    the active qubits then reads the wrong qubits (a ZZ of -1 came back as +1); compacting
    explicitly avoids that.
    """
    out = QuantumCircuit(len(local), global_phase=circuit.global_phase)
    for inst in circuit.data:
        qubits = [local[circuit.find_bit(q).index] for q in inst.qubits]
        out.append(inst.operation, qubits)
    return out


def _compact_noise(noise_model: Any, local: dict[int, int]) -> Any:
    """The device noise model restricted to the active qubits, relabelled like the circuit.

    Readout errors are left out: they are applied analytically (``energies``), and
    ``save_expectation_value`` ignores them anyway.
    """
    from qiskit_aer.noise import NoiseModel

    if noise_model._default_quantum_errors:
        msg = "device noise models with all-qubit errors are not supported"
        raise ValueError(msg)
    out = NoiseModel(basis_gates=noise_model.basis_gates)
    for gate, by_qubits in noise_model._local_quantum_errors.items():
        for qubits, error in by_qubits.items():
            if all(q in local for q in qubits):
                out.add_quantum_error(error, gate, [local[q] for q in qubits])
    return out


# -- expectation values ----------------------------------------------------------------------------


def _with_saves(compiled: Compiled) -> tuple[QuantumCircuit, list[str]]:
    """The circuit with one ``save_expectation_value`` per two-body term and per single Pauli."""
    circuit = compiled.circuit.copy()
    labels = []
    for k, (_, letter, a, b) in enumerate(compiled.terms):
        circuit.save_expectation_value(Pauli(letter * 2), [a, b], label=f"t{k}")
        labels.append(f"t{k}")
    for q in compiled.info["qubits"]:
        for letter in "XYZ":
            circuit.save_expectation_value(Pauli(letter), [q], label=f"s{q}{letter}")
            labels.append(f"s{q}{letter}")
    return circuit, labels


def expectations(
    compiled: Compiled, thetas: np.ndarray, *, noisy: bool, threads: int
) -> np.ndarray:
    """Rows: parameter vectors; columns: the saved expectation values (``_with_saves`` order)."""
    from qiskit_aer import AerSimulator

    circuit, labels = _with_saves(compiled)
    if noisy:
        options: dict[str, Any] = {"method": "density_matrix"}
        if compiled.noise_model is not None:
            options["noise_model"] = compiled.noise_model
    else:
        circuit = _strip_noise(circuit)
        options = {"method": "statevector"}
    sim = AerSimulator(max_parallel_threads=threads, **options)
    params = list(circuit.parameters)
    binds = [{p: [float(t[i]) for t in thetas] for i, p in enumerate(params)}]
    result = sim.run(circuit, parameter_binds=binds, shots=1).result()
    check_success(result)
    return np.array(
        [[float(np.real(result.data(i)[lab])) for lab in labels] for i in range(len(thetas))]
    )


def check_success(result: Any) -> None:
    """Raise unless Aer finished every experiment.

    A SIGINT (the ``timeout`` of a bounded job) can stop Aer inside a batch; it then returns
    a result whose interrupted experiment holds partial data instead of raising. Such data must
    never reach the cache.
    """
    failed = [i for i, r in enumerate(result.results) if not r.success]
    if failed or not result.success:
        msg = f"Aer did not finish experiments {failed}: {result.status}"
        raise RuntimeError(msg)


def _strip_noise(circuit: QuantumCircuit) -> QuantumCircuit:
    out = circuit.copy_empty_like()
    for inst in circuit.data:
        if inst.operation.name in ("quantum_channel", "kraus", "superop", "pauli_channel"):
            continue
        out.append(inst)
    return out


def energies(compiled: Compiled, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(E without readout error, E with readout error)`` per row of saved values."""
    nt = len(compiled.terms)
    singles = {
        (q, letter): nt + 3 * j + "XYZ".index(letter)
        for j, q in enumerate(compiled.info["qubits"])
        for letter in "XYZ"
    }
    e_gate = np.zeros(len(values))
    e_ro = np.zeros(len(values))
    for k, (coeff, letter, a, b) in enumerate(compiled.terms):
        two = values[:, k]
        e_gate += coeff * two
        (p10a, p01a), (p10b, p01b) = compiled.readout[a], compiled.readout[b]
        aa, ba = p01a - p10a, 1 - p10a - p01a
        ab, bb = p01b - p10b, 1 - p10b - p01b
        za, zb = values[:, singles[(a, letter)]], values[:, singles[(b, letter)]]
        e_ro += coeff * (ba * bb * two + aa * bb * zb + ab * ba * za + aa * ab)
    return e_gate, e_ro


# -- the run ---------------------------------------------------------------------------------------


def _stencil(theta: np.ndarray) -> np.ndarray:
    """Centre, then theta +- h e_k for every k."""
    rows = [theta]
    for k in range(theta.size):
        for sign in (1, -1):
            t = theta.copy()
            t[k] += sign * H_FD
            rows.append(t)
    return np.array(rows)


def _fd(e: np.ndarray) -> tuple[float, np.ndarray]:
    plus, minus = e[1::2], e[2::2]
    return float(e[0]), (plus - minus) / (2 * H_FD)


def ideal_gradient(reps: int, theta: np.ndarray) -> tuple[float, np.ndarray]:
    """Noise-free energy and exchange shift-rule gradient of the logical circuit."""
    cfg = _base_config(reps)
    lattice = cfg.lattice.build()
    h = heisenberg_hamiltonian(lattice).to_matrix(sparse=True)
    circuit = cfg.ansatz.build(lattice).build()

    def energy(t: np.ndarray) -> float:
        psi = Statevector(circuit.assign_parameters(t)).data
        return float(np.real(np.vdot(psi, h @ psi)))

    grad = np.empty(theta.size)
    for k in range(theta.size):
        shift = np.zeros(theta.size)
        shift[k] = np.pi / 4
        grad[k] = energy(theta + shift) - energy(theta - shift)
    return energy(theta), grad


def fold_two_qubit_gates(circuit: QuantumCircuit, factor: int) -> QuantumCircuit:
    """Local folding for ZNE: every two-qubit gate ``G`` becomes ``G (G^dag G)^((factor-1)/2)``.

    The device noise model attaches its error to every copy, so the two-qubit gate error (and the
    relaxation during it) is scaled by ``factor``; single-qubit gates are not folded.
    """
    if factor < 1 or factor % 2 == 0:
        msg = f"fold factor must be odd and >= 1, got {factor}"
        raise ValueError(msg)
    out = circuit.copy_empty_like()
    for inst in circuit.data:
        out.append(inst)
        if inst.operation.num_qubits == 2 and factor > 1:
            inverse = inst.operation.inverse()
            for _ in range((factor - 1) // 2):
                out.append(inverse, inst.qubits)
                out.append(inst.operation, inst.qubits)
    return out


DEADLINE_MARGIN_S = 300


def out_of_time() -> bool:
    """True when the job's ``DEADLINE`` (epoch seconds, set by the round driver) is near.

    Jobs stop *between* chunks instead of being interrupted inside Aer: an interrupted Aer
    batch once returned partial data for its last experiment, which was cached.
    """
    deadline = os.environ.get("DEADLINE")
    return deadline is not None and time.time() > float(deadline) - DEADLINE_MARGIN_S


def bad_rows(gate_energies: np.ndarray, tolerance: float = 0.05) -> list[int]:
    """Stencil rows inconsistent with the centre: ``|E(theta +- h e_k) - E(theta)| <= |g| h``.

    With ``h = 1e-3`` a genuine row differs from the centre by ~1e-3; a truncated or otherwise
    corrupt row by ~1.
    """
    return [
        i
        for i in range(1, len(gate_energies))
        if abs(gate_energies[i] - gate_energies[0]) > tolerance
    ]


def _key(label: str, fold: int) -> str:
    return label if fold == 1 else f"{label}_fold{fold}"


def rs_energies(compiled: Compiled, thetas: np.ndarray, threads: int) -> np.ndarray:
    """Infinite-shot readout + symmetry energies (the M2 estimator) per row of ``thetas``.

    Needs the full outcome distributions, so one density matrix per row (no saved expectation
    values): per uniform basis the distribution, the device's per-qubit readout confusion, then
    the correct-then-filter ratio estimator of ``LogicalNoiseEnergy``.
    """
    from m2_exact_limits import basis_probabilities, basis_statistic
    from qiskit_aer import AerSimulator

    from kagome_vqe.mitigation.counts import _apply_inverse

    width = len(compiled.info["qubits"])
    q = np.array([compiled.readout[i] for i in range(width)])
    confusion = np.array([[[1 - a, b], [a, 1 - b]] for a, b in q])
    inverse = np.linalg.inv(confusion)
    bits = ((np.arange(2**width)[:, None] >> np.arange(width)) & 1).astype(float)
    z = 1 - 2 * bits
    diagonals = {
        b: sum(
            (c * z[:, i] * z[:, j] for c, letter, i, j in compiled.terms if letter == b),
            start=np.zeros(2**width),
        )
        for b in "XYZ"
    }
    circuit = compiled.circuit.copy()
    circuit.save_density_matrix()
    options: dict[str, Any] = {"method": "density_matrix"}
    if compiled.noise_model is not None:
        options["noise_model"] = compiled.noise_model
    sim = AerSimulator(max_parallel_threads=threads, **options)
    out = []
    for theta in thetas:
        result = sim.run(circuit.assign_parameters(theta)).result()
        check_success(result)
        rho = np.asarray(result.data()["density_matrix"], dtype=complex)
        total = 0.0
        for b in "XYZ":
            probs = np.clip(basis_probabilities(rho, b * width), 0, None)
            read = _apply_inverse(probs, confusion)
            total += basis_statistic(read, diagonals[b], symmetry=True, inverses=inverse, n=width)[
                0
            ]
        out.append(total)
    return np.asarray(out)


def run_point_rs(model: str, point: Point, threads: int) -> Path:
    """The readout + symmetry stencil of (model, point), chunk by chunk (resumable)."""
    out_dir = CACHE / model
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{point.label}_rs.npz"
    if final.exists():
        return final
    compiled = compile_model(model, point.reps)
    fingerprint = circuit_fingerprint(compiled.circuit)
    stencil = _stencil(point.theta)
    chunks = []
    for start in range(0, len(stencil), CHUNK):
        part = out_dir / f"{point.label}_rs.{fingerprint}.part{start:04d}.npy"
        if not part.exists():
            if out_of_time():
                log.info("%s %s rs: deadline, stopping before rows %d", model, point.label, start)
                sys.exit(0)
            vals = rs_energies(compiled, stencil[start : start + CHUNK], threads)
            np.save(part.with_suffix(".tmp.npy"), vals)
            part.with_suffix(".tmp.npy").rename(part)
            log.info("%s %s rs rows %d-%d done", model, point.label, start, start + len(vals) - 1)
        chunks.append(np.load(part))
    values = np.concatenate(chunks)
    if bad := bad_rows(values):
        msg = f"{model} {point.label} rs: rows {bad} inconsistent with the centre"
        raise RuntimeError(msg)
    np.savez(final, rs=values, fingerprint=fingerprint)
    for part in out_dir.glob(f"{point.label}_rs.*part*.npy"):
        part.unlink()
    return final


def run_point(model: str, point: Point, threads: int, fold: int = 1) -> Path:
    """Fill the cache file of (model, point[, fold]), chunk by chunk."""
    out_dir = CACHE / model
    out_dir.mkdir(parents=True, exist_ok=True)
    key = _key(point.label, fold)
    final = out_dir / f"{key}.npz"
    if final.exists():
        log.info("%s %s: cached", model, key)
        return final
    tick = time.perf_counter()
    compiled = compile_model(model, point.reps)
    compiled.info["base_fingerprint"] = circuit_fingerprint(compiled.circuit)
    if fold > 1:
        if compiled.noise_model is None:
            msg = "folding is for the device models (the layer model scales probabilities)"
            raise ValueError(msg)
        compiled.circuit = fold_two_qubit_gates(compiled.circuit, fold)
        compiled.info["fold"] = fold
    compiled.info["fingerprint"] = circuit_fingerprint(compiled.circuit)
    log.info("%s p=%d compiled: %s", model, point.reps, compiled.info)
    stencil = _stencil(point.theta)
    chunks = []
    for start in range(0, len(stencil), CHUNK):
        part = out_dir / f"{key}.{compiled.info['fingerprint']}.part{start:04d}.npy"
        if not part.exists():
            if out_of_time():
                log.info("%s %s: deadline, stopping before rows %d", model, key, start)
                sys.exit(0)
            t0 = time.perf_counter()
            vals = expectations(
                compiled, stencil[start : start + CHUNK], noisy=True, threads=threads
            )
            np.save(part.with_suffix(".tmp.npy"), vals)
            part.with_suffix(".tmp.npy").rename(part)
            log.info(
                "%s %s rows %d-%d: %.1f s/row",
                model,
                key,
                start,
                start + len(vals) - 1,
                (time.perf_counter() - t0) / len(vals),
            )
        chunks.append(np.load(part))
    noisy = np.vstack(chunks)
    gate, _ = energies(compiled, noisy)
    if bad := bad_rows(gate):
        for start in sorted({CHUNK * (i // CHUNK) for i in bad}):
            (out_dir / f"{key}.{compiled.info['fingerprint']}.part{start:04d}.npy").unlink()
        msg = f"{model} {key}: rows {bad} inconsistent with the centre; chunks removed, rerun"
        raise RuntimeError(msg)
    clean = expectations(compiled, stencil, noisy=False, threads=threads)
    e_ideal, g_ideal = ideal_gradient(point.reps, point.theta)
    np.savez(
        final.with_suffix(".tmp.npz"),
        theta=point.theta,
        noisy_values=noisy,
        clean_values=clean,
        e_ideal=e_ideal,
        g_ideal=g_ideal,
        info=np.array(repr(compiled.info)),
        terms=np.array(compiled.terms, dtype=object),
        readout=np.array(sorted(compiled.readout.items()), dtype=object),
    )
    final.with_suffix(".tmp.npz").rename(final)
    for part in out_dir.glob(f"{key}.*part*.npy"):
        part.unlink()
    log.info("%s %s done in %.0f s", model, key, time.perf_counter() - tick)
    return final


# -- report ----------------------------------------------------------------------------------------


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def analyse(model: str, point: Point) -> dict[str, Any] | None:
    """One row of ``points.csv`` from the cache, or ``None`` if not computed."""
    path = CACHE / model / f"{point.label}.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    terms = [tuple(t) for t in z["terms"]]
    readout = {int(q): tuple(v) for q, v in z["readout"]}
    info = ast.literal_eval(str(z["info"]))
    compiled = Compiled(QuantumCircuit(1), terms, readout, None, info)
    e_gate, e_ro = energies(compiled, z["noisy_values"])
    e_clean, _ = energies(compiled, z["clean_values"])
    en, gn = _fd(e_gate)
    er, gr = _fd(e_ro)
    ec, gc = _fd(e_clean)
    ei, gi = float(z["e_ideal"]), z["g_ideal"]
    row: dict[str, Any] = {
        "model": model,
        "point": point.label,
        "reps": point.reps,
        "e_ideal": ei,
        "e_noisy": en,
        "e_noisy_ro": er,
        "bias": en - ei,
        "bias_ro": er - ei,
        "damping": en / ei,
        "damping_ro": er / ei,
        "g_ideal_norm": float(np.linalg.norm(gi)),
        "g_noisy_norm": float(np.linalg.norm(gn)),
        "g_noisy_ro_norm": float(np.linalg.norm(gr)),
        "cos": _cos(gn, gi),
        "cos_ro": _cos(gr, gi),
        "fd_check": float(np.max(np.abs(gc - gi))),
        "isa_energy_check": float(abs(ec - ei)),
        "two_qubit_gates": info.get("two_qubit_gates"),
        "mapping": info.get("mapping"),
    }
    rs_path = CACHE / model / f"{point.label}_rs.npz"
    if rs_path.exists():
        e_rs, g_rs = _fd(np.load(rs_path)["rs"])
        row["e_rs"], row["bias_rs"], row["cos_rs"] = e_rs, e_rs - ei, _cos(g_rs, gi)
    for suffix, g in (("", gn), ("_ro", gr)):
        c = float(g @ gi / (gi @ gi))
        resid = g - c * gi
        row[f"scale{suffix}"] = c
        row[f"resid{suffix}"] = float(np.linalg.norm(resid))
        row[f"resid_over_gideal{suffix}"] = float(np.linalg.norm(resid) / np.linalg.norm(gi))
        # gradient of the bias, relative to the true gradient: is E_noisy - D E_ideal flat?
        d = row["damping" + suffix]
        row[f"bias_grad_over_gideal{suffix}"] = float(
            np.linalg.norm(g - d * gi) / np.linalg.norm(gi)
        )
    return row


FOLDS = (1, 3, 5)


def _gate_energies(model: str, key: str) -> np.ndarray | None:
    """Gate-noise energies (perfect readout correction) of the cached stencil, or ``None``."""
    path = CACHE / model / f"{key}.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    terms = [tuple(t) for t in z["terms"]]
    readout = {int(q): tuple(v) for q, v in z["readout"]}
    compiled = Compiled(QuantumCircuit(1), terms, readout, None, {"qubits": sorted(readout)})
    return energies(compiled, z["noisy_values"])[0]


def richardson_weights(scales: tuple[float, ...], degree: int) -> np.ndarray:
    """Weights ``w`` with ``f(0) ~ sum_i w_i f(scales_i)`` for a least-squares polynomial fit.

    Linear in the data, so the extrapolated gradient is the same combination of gradients.
    """
    x = np.asarray(scales, dtype=float)
    vander = np.vander(x, degree + 1, increasing=True)
    return np.linalg.pinv(vander)[0]


def analyse_zne(model: str, point: Point) -> dict[str, Any] | None:
    """ZNE by two-qubit gate folding (1, 3, 5), exact: direction of the extrapolated gradient."""
    data = [_gate_energies(model, _key(point.label, f)) for f in FOLDS]
    if any(d is None for d in data):
        return None
    layouts = set()
    for f in FOLDS:
        z = np.load(CACHE / model / f"{_key(point.label, f)}.npz", allow_pickle=True)
        info = ast.literal_eval(str(z["info"]))
        layouts.add(
            (
                tuple(info.get("physical_qubits") or ()),
                str(info.get("mapping")),
                info.get("base_fingerprint"),
            )
        )
    if len(layouts) > 1:
        log.warning(
            "%s %s: folds compiled differently, ZNE skipped: %s", model, point.label, layouts
        )
        return None
    base = np.load(CACHE / model / f"{point.label}.npz", allow_pickle=True)
    ei, gi = float(base["e_ideal"]), base["g_ideal"]
    values, grads = zip(*(_fd(d) for d in data if d is not None), strict=True)
    row: dict[str, Any] = {"model": model, "point": point.label, "e_ideal": ei}
    for f, e, g in zip(FOLDS, values, grads, strict=True):
        row[f"bias_x{f}"] = e - ei
        row[f"cos_x{f}"] = _cos(g, gi)
    for name, degree in (("linear", 1), ("quadratic", 2)):
        w = richardson_weights(FOLDS, degree)
        e = float(w @ np.asarray(values))
        g = w @ np.asarray(grads)
        row[f"bias_zne_{name}"] = e - ei
        row[f"cos_zne_{name}"] = _cos(g, gi)
        row[f"norm_ratio_zne_{name}"] = float(np.linalg.norm(g) / np.linalg.norm(gi))
        row[f"err_over_gideal_zne_{name}"] = float(np.linalg.norm(g - gi) / np.linalg.norm(gi))
    row["err_over_gideal_x1"] = float(np.linalg.norm(grads[0] - gi) / np.linalg.norm(gi))
    # noise amplification: ZNE multiplies the per-shot variance by sum(w_i^2) at equal shots
    row["variance_factor_linear"] = float(np.sum(richardson_weights(FOLDS, 1) ** 2))
    row["variance_factor_quadratic"] = float(np.sum(richardson_weights(FOLDS, 2) ** 2))
    return row


_NAMES = {
    "layer": "PRB 106 layer\n1e-3 (control)",
    "nighthawk_prb105": "Nighthawk\nPRB 105 mapping",
    "kingston_ring": "Kingston\nbest hexagon",
    "kingston_median": "Kingston\nmedian hexagon",
    "guadalupe_ring": "Guadalupe\nM2 ring",
}


def plot_direction(points: pd.DataFrame, zne: pd.DataFrame | None, path: Path) -> None:
    """Cosine with the noise-free gradient at the seed-1 iterate 10, per model and estimator."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    sel = points[points["point"] == "gd_p2_s1_it10"].set_index("model")
    models = [m for m in _NAMES if m in sel.index]
    series = [
        ("gate noise (perfect readout correction)", sel.get("cos")),
        ("unmitigated (with readout error)", sel.get("cos_ro")),
        ("readout + symmetry", sel.get("cos_rs")),
    ]
    if zne is not None and len(zne):
        z = zne[zne["point"] == "gd_p2_s1_it10"].set_index("model")["cos_zne_quadratic"]
        series.append(("quadratic ZNE (gate noise)", z))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.8 / len(series)
    for i, (label, values) in enumerate(series):
        if values is None:
            continue
        ys = [float(values.get(m, np.nan)) for m in models]
        xs = np.arange(len(models)) + (i - (len(series) - 1) / 2) * width
        ax.bar(xs, ys, width=width, label=label)
    ax.axhline(0.95, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(len(models)), [_NAMES[m] for m in models], fontsize=8)
    ax.set_ylabel("cos(noisy gradient, noise-free gradient)")
    ax.set_ylim(0, 1.02)
    ax.set_title("2x6, p = 2, GD iterate 10 (seed 1); exact, no shots")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _md(frame: pd.DataFrame) -> str:
    """A Markdown table (pandas' ``to_markdown`` needs ``tabulate``, not a dependency)."""

    def fmt(v: Any) -> str:
        if isinstance(v, float | np.floating):
            return f"{v:.4g}"
        return str(v)

    head = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def report(out: Path) -> None:
    """Collect every cached (model, point) and write the tables."""
    pts = points()
    rows = [r for m in MODELS for p in pts.values() if (r := analyse(m, p)) is not None]
    frame = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "points.csv", index=False)
    cols = [
        "model",
        "point",
        "e_ideal",
        "bias",
        "damping",
        "g_ideal_norm",
        "g_noisy_norm",
        "cos",
        "cos_ro",
        "cos_rs",
        "scale",
        "resid",
        "fd_check",
    ]
    cols = [c for c in cols if c in frame]
    lines = ["# Noise direction (exact)", "", _md(frame[cols])]
    summary = (
        frame.groupby("model")
        .agg(
            n=("cos", "size"),
            cos_min=("cos", "min"),
            cos_ro_min=("cos_ro", "min"),
            damping_mean=("damping", "mean"),
            damping_std=("damping", "std"),
            bias_mean=("bias", "mean"),
            bias_std=("bias", "std"),
        )
        .reset_index()
    )
    summary.to_csv(out / "summary.csv", index=False)
    lines += ["", "## Per model", "", _md(summary)]
    zne = [r for m in MODELS for p in pts.values() if (r := analyse_zne(m, p)) is not None]
    plot_direction(frame, pd.DataFrame(zne) if zne else None, out / "noise_direction.png")
    if zne:
        zf = pd.DataFrame(zne)
        zf.to_csv(out / "zne.csv", index=False)
        zcols = [
            "model",
            "point",
            "cos_x1",
            "cos_x3",
            "cos_x5",
            "cos_zne_linear",
            "cos_zne_quadratic",
            "err_over_gideal_x1",
            "err_over_gideal_zne_linear",
            "err_over_gideal_zne_quadratic",
            "bias_x1",
            "bias_zne_linear",
            "bias_zne_quadratic",
        ]
        lines += ["", "## ZNE by two-qubit gate folding (exact)", "", _md(zf[zcols])]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    sys.stdout.write("\n".join(lines) + "\n")


def plan() -> None:
    """Points, models and the number of density matrices."""
    pts = points()
    total = 0
    for p in pts.values():
        n = 2 * p.theta.size + 1
        total += n
        sys.stdout.write(
            f"{p.label:18s} p={p.reps} params={p.theta.size:3d} DMs/model={n:4d}  {p.source}\n"
        )
    sys.stdout.write(f"density matrices per model: {total}; models: {len(MODELS)}\n")


def main() -> None:
    """CLI."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    r = sub.add_parser("run")
    r.add_argument("model", choices=MODELS)
    r.add_argument("points", nargs="+")
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--fold", type=int, default=1, help="ZNE: fold two-qubit gates (3, 5)")
    r.add_argument("--rs", action="store_true", help="readout + symmetry energies instead")
    rep = sub.add_parser("report")
    rep.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.cmd == "plan":
        plan()
    elif args.cmd == "run":
        pts = points()
        for label in args.points:
            if args.rs:
                run_point_rs(args.model, pts[label], args.threads)
            else:
                run_point(args.model, pts[label], args.threads, args.fold)
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        report(args.out or ROOT / "results" / f"{stamp}-noise_direction")


if __name__ == "__main__":
    main()
