"""Hardware mapping: synthesis, SWAP routing, permutations, paper constructions, placement."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from kagome_vqe.ansatz import GraphHVAnsatz
from kagome_vqe.ansatz.unshare import unshare_parameters
from kagome_vqe.hamiltonian import heisenberg_hamiltonian
from kagome_vqe.kagome_lattices import (
    custom_lattice,
    kagome_bm_rectangle,
    kagome_kvqe20,
    kagome_patch,
    kagome_periodic,
)
from kagome_vqe.lattice import Lattice, kagome_unit_cell
from kagome_vqe.mapping import (
    MappingError,
    exchange_cx3,
    logical_ops,
    map_circuit,
    presynthesise_exchange,
    verify_routed,
)
from kagome_vqe.mapping.placement import backend_coupling_graph, place_on_backend
from kagome_vqe.schedule import ExchangeSchedule, make_schedule
from kagome_vqe.transpile import compile_mapping


def _exchange(ax: float, ay: float, az: float) -> Operator:
    qc = QuantumCircuit(2)
    qc.rzz(az, 0, 1)
    qc.ryy(ay, 0, 1)
    qc.rxx(ax, 0, 1)
    return Operator(qc)


def _setup(
    lattice: Lattice, schedule_kind: str, reps: int = 2, *, include_h0: bool = True
) -> tuple[GraphHVAnsatz, ExchangeSchedule, QuantumCircuit]:
    schedule = make_schedule(lattice, schedule_kind, dimers=lattice.dimers)  # type: ignore[arg-type]
    ansatz = GraphHVAnsatz(lattice, reps, schedule=schedule, include_h0=include_h0)
    return ansatz, schedule, ansatz.build()


# -- synthesis ------------------------------------------------------------------------------


def test_exchange_cx3_is_exact(rng: np.random.Generator) -> None:
    for _ in range(10):
        ax, ay, az = rng.uniform(-np.pi, np.pi, 3)
        qc = QuantumCircuit(2)
        exchange_cx3(qc, ax, ay, az, 0, 1)
        assert qc.count_ops()["cx"] == 3
        assert Operator(qc).equiv(_exchange(ax, ay, az))


def test_swap_merges_into_exchange(rng: np.random.Generator) -> None:
    """G(theta) SWAP == G(theta + pi/2) up to a global phase."""
    theta = rng.uniform(-np.pi, np.pi, 3)
    qc = QuantumCircuit(2)
    qc.rzz(theta[2], 0, 1)
    qc.ryy(theta[1], 0, 1)
    qc.rxx(theta[0], 0, 1)
    qc.swap(0, 1)
    assert Operator(qc).equiv(_exchange(*(theta + np.pi / 2)))


def test_logical_ops_group_exchange_blocks() -> None:
    circuit = GraphHVAnsatz(kagome_unit_cell(), 1).build()
    ops = logical_ops(circuit)
    assert sum(op.kind == "exchange" for op in ops) == 18
    assert sum(op.kind == "two" for op in ops) == 6  # singlet CNOTs


def test_presynthesis_preserves_the_state(rng: np.random.Generator) -> None:
    circuit = GraphHVAnsatz(kagome_bm_rectangle(2, 4), 2, include_h0=True).build()
    values = rng.uniform(-1, 1, circuit.num_parameters)
    a = Statevector(circuit.assign_parameters(values))
    b = Statevector(presynthesise_exchange(circuit).assign_parameters(values))
    assert abs(a.inner(b)) ** 2 == pytest.approx(1.0)


# -- routed circuits equal logical circuits -------------------------------------------------

CASES = [
    ("star", kagome_unit_cell(), "greedy"),
    ("star", kagome_unit_cell(), "prb106"),
    ("star_identity_order", kagome_unit_cell(site_order=None), "prb105"),
    ("bm2x4", kagome_bm_rectangle(2, 4), "prb105"),
    ("bm2x4", kagome_bm_rectangle(2, 4), "prb105_fig3"),
    ("bm2x6", kagome_bm_rectangle(2, 6), "prb105"),
    ("patch3x4", kagome_patch(3, 4), "greedy"),
]


@pytest.mark.parametrize(("label", "lattice", "schedule_kind"), CASES)
@pytest.mark.parametrize("strategy", ["prb105", "prb106_grid"])
@pytest.mark.parametrize(("restore", "merge"), [(False, True), (True, False)])
def test_routed_state_equals_logical_state(
    label: str,
    lattice: Lattice,
    schedule_kind: str,
    strategy: str,
    restore: bool,
    merge: bool,
    rng: np.random.Generator,
) -> None:
    del label
    _, schedule, circuit = _setup(lattice, schedule_kind)
    result = map_circuit(
        strategy,  # type: ignore[arg-type]
        lattice,
        circuit,
        heisenberg_hamiltonian(lattice),
        schedule=schedule,
        restore_at_end=restore,
        merge_swaps=merge,
    )
    assert result.routed is not None
    if result.circuit.num_qubits > 18:
        pytest.skip("too wide for a dense statevector check")
    values = rng.uniform(-np.pi, np.pi, circuit.num_parameters)
    fidelity = verify_routed(circuit, result.circuit, result.routed, values)  # checks aux |0>
    assert fidelity == pytest.approx(1.0, abs=1e-9)
    if restore:
        assert result.final_qubits == result.initial_qubits


def test_routed_energy_uses_permuted_observable(rng: np.random.Generator) -> None:
    lattice = kagome_bm_rectangle(2, 6)
    hamiltonian = heisenberg_hamiltonian(lattice)
    _, schedule, circuit = _setup(lattice, "prb105_fig3")  # H4, H5 last: SWAPs left over
    result = map_circuit("prb105", lattice, circuit, hamiltonian, schedule=schedule)
    assert result.final_qubits != result.initial_qubits  # a genuine permutation is left over
    values = rng.uniform(-1, 1, circuit.num_parameters)
    logical = Statevector(circuit.assign_parameters(values)).expectation_value(hamiltonian).real
    routed = Statevector(result.circuit.assign_parameters(values)).expectation_value(
        result.observable
    )
    assert routed.real == pytest.approx(logical)


def test_parameter_shift_circuit_routes(rng: np.random.Generator) -> None:
    """Unshared rotations (distinct XX/YY/ZZ angles) still form exchange blocks and route."""
    lattice = kagome_bm_rectangle(2, 4)
    _, schedule, circuit = _setup(lattice, "prb105")
    unshared, _ = unshare_parameters(circuit)
    result = map_circuit(
        "prb106_grid", lattice, unshared, heisenberg_hamiltonian(lattice), schedule=schedule
    )
    assert result.routed is not None
    values = rng.uniform(-1, 1, unshared.num_parameters)
    assert verify_routed(unshared, result.circuit, result.routed, values) == pytest.approx(1.0)


# -- the paper constructions ----------------------------------------------------------------


def _depths(lattice: Lattice, schedule_kind: str, strategy: str, reps: range) -> list[int]:
    depths = []
    for p in reps:
        _, schedule, circuit = _setup(lattice, schedule_kind, p, include_h0=False)
        result = map_circuit(
            strategy,  # type: ignore[arg-type]
            lattice,
            circuit,
            heisenberg_hamiltonian(lattice),
            schedule=schedule,
            merge_swaps=False,
        )
        assert result.status == "paper_exact"
        depths.append(result.stats["block_2q_depth"])
    return depths


def test_prb106_kvqe20_is_paper_exact() -> None:
    """PRB 106: 24 grid qubits (4 stations), 30 HEIS + 16 SWAP gates and depth 6 per cycle."""
    lattice = kagome_kvqe20()
    _, schedule, circuit = _setup(lattice, "prb106", 3, include_h0=False)
    result = map_circuit(
        "prb106_grid", lattice, circuit, heisenberg_hamiltonian(lattice), schedule=schedule
    )
    assert result.status == "paper_exact"
    assert result.stats["physical_qubits"] == 24
    assert len(result.auxiliary_qubits) == 4
    assert result.stats["exchange_blocks"] == 30 * 3
    depths = _depths(lattice, "prb106", "prb106_grid", range(1, 5))
    assert np.diff(depths).tolist() == [6, 6, 6]
    swaps = []
    for p in (2, 3):
        _, sched, circ = _setup(lattice, "prb106", p, include_h0=False)
        res = map_circuit(
            "prb106_grid", lattice, circ, heisenberg_hamiltonian(lattice), schedule=sched
        )
        swaps.append(res.stats["routing_swaps"])
    assert swaps[1] - swaps[0] == 16


@pytest.mark.parametrize(("rows", "cols"), [(3, 8), (3, 6), (2, 10)])
def test_prb105_rectangles_are_paper_exact(rows: int, cols: int) -> None:
    """PRB 105: no auxiliary qubits and 7 two-qubit layers per cycle (Fig. 3)."""
    lattice = kagome_bm_rectangle(rows, cols)
    _, schedule, circuit = _setup(lattice, "prb105", 1, include_h0=False)
    result = map_circuit(
        "prb105", lattice, circuit, heisenberg_hamiltonian(lattice), schedule=schedule
    )
    assert result.auxiliary_qubits == ()
    assert result.stats["physical_qubits"] == rows * cols
    depths = _depths(lattice, "prb105", "prb105", range(2, 5))
    assert np.diff(depths).tolist() == [7, 7]


def test_other_lattices_are_only_paper_inspired() -> None:
    lattice = kagome_unit_cell()
    _, schedule, circuit = _setup(lattice, "greedy")
    for strategy in ("prb105", "prb106_grid"):
        result = map_circuit(
            strategy,
            lattice,
            circuit,
            heisenberg_hamiltonian(lattice),
            schedule=schedule,
        )
        assert result.status == "paper_inspired"
        assert result.notes


# -- unsupported configurations -------------------------------------------------------------


def test_periodic_lattice_is_rejected_unless_fallback_allowed() -> None:
    lattice = kagome_periodic(2, 2)
    circuit = GraphHVAnsatz(lattice, 1).build()
    hamiltonian = heisenberg_hamiltonian(lattice)
    with pytest.raises(MappingError, match="periodic"):
        map_circuit("prb106_grid", lattice, circuit, hamiltonian)
    result = map_circuit("prb106_grid", lattice, circuit, hamiltonian, fallback="generic")
    assert (result.used, result.status) == ("generic", "generic_fallback")
    assert "fallback" in result.notes[0]


def test_lattice_without_geometry_is_rejected() -> None:
    lattice = custom_lattice(4, [[0, 1], [1, 2], [2, 3], [3, 0]])
    circuit = GraphHVAnsatz(lattice, 1).build()
    with pytest.raises(MappingError, match="coordinates"):
        map_circuit("prb105", lattice, circuit, heisenberg_hamiltonian(lattice))


def test_non_local_gate_is_rejected_by_geometric_router() -> None:
    lattice = kagome_bm_rectangle(2, 4)
    circuit = QuantumCircuit(8)
    circuit.cx(0, 7)
    with pytest.raises(MappingError, match="neighbours"):
        map_circuit("prb106_grid", lattice, circuit, heisenberg_hamiltonian(lattice))


# -- hardware placement ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def nighthawk() -> object:
    from qiskit_ibm_runtime.fake_provider import FakeNighthawk

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return FakeNighthawk()


@pytest.fixture(scope="module")
def boston() -> object:
    from qiskit_ibm_runtime.fake_provider import FakeBoston

    return FakeBoston()


def test_placement_respects_coupling_map(nighthawk: object) -> None:
    lattice = kagome_kvqe20()
    _, schedule, circuit = _setup(lattice, "prb106", 1)
    result = map_circuit(
        "prb106_grid", lattice, circuit, heisenberg_hamiltonian(lattice), schedule=schedule
    )
    placement = place_on_backend(result.circuit.num_qubits, sorted(result.coupling), nighthawk)
    _, errors = backend_coupling_graph(nighthawk)
    phys = placement.physical
    assert len(set(phys)) == len(phys)
    for a, b in result.coupling:
        assert (min(phys[a], phys[b]), max(phys[a], phys[b])) in errors
    again = place_on_backend(result.circuit.num_qubits, sorted(result.coupling), nighthawk)
    assert again.physical == phys  # deterministic


def test_explicit_layout_is_validated(nighthawk: object) -> None:
    with pytest.raises(MappingError, match="lacks hardware couplers"):
        place_on_backend(2, [(0, 1)], nighthawk, explicit=[0, 99])
    ok = place_on_backend(2, [(0, 1)], nighthawk, explicit=[0, 1])
    assert ok.method == "explicit"


def test_grid_does_not_fit_heavy_hex(boston: object) -> None:
    lattice = kagome_bm_rectangle(2, 4)
    _, schedule, circuit = _setup(lattice, "prb105", 1)
    result = map_circuit(
        "prb105", lattice, circuit, heisenberg_hamiltonian(lattice), schedule=schedule
    )
    with pytest.raises(MappingError, match="no embedding"):
        place_on_backend(result.circuit.num_qubits, sorted(result.coupling), boston)


@pytest.mark.parametrize("strategy", ["prb105", "prb106_grid", "generic"])
def test_isa_circuit_on_nighthawk_reproduces_logical_energy(
    strategy: str, nighthawk: object, rng: np.random.Generator
) -> None:
    from qiskit_aer.primitives import EstimatorV2

    lattice = kagome_bm_rectangle(2, 4)
    hamiltonian = heisenberg_hamiltonian(lattice)
    _, schedule, circuit = _setup(lattice, "prb105", 1)
    result = map_circuit(
        strategy,  # type: ignore[arg-type]
        lattice,
        circuit,
        hamiltonian,
        schedule=schedule,
    )
    problem = compile_mapping(result, hamiltonian, nighthawk, seed_transpiler=7)
    assert set(problem.circuit.count_ops()) <= {"cz", "rz", "sx", "x", "barrier"}
    assert problem.mapping is not None
    assert problem.mapping["status"] in ("paper_exact", "paper_inspired", "generic")
    values = rng.uniform(-1, 1, circuit.num_parameters)
    logical = Statevector(circuit.assign_parameters(values)).expectation_value(hamiltonian).real
    estimator = EstimatorV2(options={"backend_options": {"method": "statevector"}})
    isa = estimator.run([(problem.circuit, problem.observable, values)]).result()[0].data.evs
    assert float(isa) == pytest.approx(logical, abs=1e-8)
    # the logical->physical map used for counts-level mitigation is consistent with the observable
    assert problem.physical_qubits is not None
    assert problem.logical_observable is not None
    assert (
        problem.logical_observable.apply_layout(
            list(problem.physical_qubits), problem.circuit.num_qubits
        ).simplify()
        == problem.observable.simplify()
    )
