"""Schedules and the graph-based HVA: structure, parameter counts, symmetry, physics."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from kagome_vqe.ansatz import GraphHVAnsatz, HVAnsatz
from kagome_vqe.ansatz.unshare import unshare_parameters
from kagome_vqe.hamiltonian import heisenberg_hamiltonian
from kagome_vqe.kagome_lattices import (
    KVQE20_COORDS,
    custom_lattice,
    kagome_bm_rectangle,
    kagome_kvqe20,
    kagome_periodic,
)
from kagome_vqe.lattice import Lattice, kagome_unit_cell
from kagome_vqe.mapping import logical_ops
from kagome_vqe.schedule import (
    ExchangeSchedule,
    explicit_schedule,
    greedy_schedule,
    make_schedule,
    prb105_schedule,
    prb106_schedule,
)

LATTICES = {
    "star": kagome_unit_cell(),
    "star_identity": kagome_unit_cell(site_order=None),
    "bm2x4": kagome_bm_rectangle(2, 4),
    "torus2x2": kagome_periodic(2, 2),
}


def _bonds(lattice: Lattice) -> set[frozenset[int]]:
    return {frozenset((a, b)) for a, b, _ in lattice.edges}


def _total_spin_squared(n: int) -> SparsePauliOp:
    """``S_tot^2`` with ``S = sigma / 2``: (3n/4) + (1/2) sum_{i<j} (XX + YY + ZZ)."""
    terms: list[tuple[str, list[int], float]] = [("", [], 3 * n / 4)]
    terms += [
        (p, [i, j], 0.5) for i in range(n) for j in range(i + 1, n) for p in ("XX", "YY", "ZZ")
    ]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n).simplify()


# -- schedules ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(LATTICES))
def test_greedy_schedule_is_a_minimal_edge_colouring(name: str) -> None:
    lattice = LATTICES[name]
    schedule = greedy_schedule(lattice, lattice.dimers)
    schedule.validate(lattice)
    assert schedule.num_layers == max(lattice.degree(s) for s in range(lattice.num_sites))
    assert greedy_schedule(lattice, lattice.dimers) == schedule  # deterministic


def test_prb106_schedule_reproduces_the_paper_layers() -> None:
    """PRB 106 Fig. 2 panels (1)-(6), transcribed by hand as grid-coordinate pairs."""
    paper = [
        [((0, 1), (0, 2)), ((2, 1), (2, 2)), ((2, 3), (2, 4)), ((4, 1), (4, 2)), ((4, 3), (4, 4))],
        [((0, 1), (1, 2)), ((0, 3), (1, 4)), ((2, 1), (3, 2)), ((2, 3), (3, 4))],
        [((0, 0), (0, 1)), ((0, 2), (0, 3)), ((2, 0), (2, 1)), ((2, 2), (2, 3)),
         ((4, 0), (4, 1)), ((4, 2), (4, 3))],
        [((0, 0), (1, 0)), ((0, 2), (1, 2)), ((2, 0), (3, 0)), ((2, 2), (3, 2)), ((2, 4), (3, 4))],
        [((1, 0), (2, 1)), ((1, 2), (2, 3)), ((3, 0), (4, 1)), ((3, 2), (4, 3))],
        [((1, 0), (2, 0)), ((1, 2), (2, 2)), ((1, 4), (2, 4)), ((3, 0), (4, 0)),
         ((3, 2), (4, 2)), ((3, 4), (4, 4))],
    ]  # fmt: skip
    schedule = prb106_schedule(kagome_kvqe20())
    index = {c: i for i, c in enumerate(KVQE20_COORDS)}
    expected = [{frozenset((index[p], index[q])) for p, q in layer} for layer in paper]
    assert [{frozenset(b) for b in layer} for layer in schedule.layers] == expected


def test_prb105_schedule_orders() -> None:
    lattice = kagome_bm_rectangle(3, 8)
    eq4 = prb105_schedule(lattice)
    fig3 = prb105_schedule(lattice, order="fig3")
    assert eq4.labels == ("H4", "H5", "H1", "H2", "H3")
    assert fig3.labels == ("H1", "H2", "H3", "H4", "H5")
    assert {frozenset(b) for b in eq4.layers[2]} == {frozenset(d) for d in lattice.dimers or ()}
    eq4.validate(lattice)
    fig3.validate(lattice)


def test_geometric_schedules_reject_periodic_lattices() -> None:
    with pytest.raises(ValueError, match="open kagome patch"):
        prb106_schedule(kagome_periodic(2, 2))
    with pytest.raises(ValueError, match="open kagome patch"):
        prb105_schedule(custom_lattice(4, [[0, 1], [1, 2], [2, 3], [3, 0]]))


def test_explicit_schedule_validation() -> None:
    lattice = custom_lattice(4, [[0, 1], [1, 2], [2, 3], [3, 0]])
    good = explicit_schedule(lattice, [[[0, 1], [2, 3]], [[1, 2], [3, 0]]])
    assert good.bonds == ((0, 1), (2, 3), (0, 3), (1, 2))
    with pytest.raises(ValueError, match="not a matching"):
        explicit_schedule(lattice, [[[0, 1], [1, 2]], [[2, 3], [3, 0]]])
    with pytest.raises(ValueError, match="misses"):
        explicit_schedule(lattice, [[[0, 1], [2, 3]]])
    with pytest.raises(ValueError, match="more than once"):
        explicit_schedule(lattice, [[[0, 1], [2, 3]], [[1, 2], [3, 0]], [[1, 0]]])
    with pytest.raises(ValueError, match="not a bond"):
        explicit_schedule(lattice, [[[0, 2]]])


def test_reordered_schedule() -> None:
    schedule = prb105_schedule(kagome_bm_rectangle(2, 6), order="fig3")
    assert (
        schedule.reordered([3, 4, 0, 1, 2]).labels
        == prb105_schedule(kagome_bm_rectangle(2, 6)).labels
    )
    with pytest.raises(ValueError, match="permutation"):
        schedule.reordered([0, 0, 1, 2, 3])


# -- structure ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(LATTICES))
@pytest.mark.parametrize("reps", [1, 3])
def test_opg_parameter_counts(name: str, reps: int) -> None:
    lattice = LATTICES[name]
    plain = GraphHVAnsatz(lattice, reps)
    with_h0 = GraphHVAnsatz(lattice, reps, include_h0=True)
    assert plain.num_parameters == plain.build().num_parameters == reps * lattice.num_bonds
    assert (
        with_h0.num_parameters == with_h0.build().num_parameters == reps * (lattice.num_bonds + 1)
    )


def test_per_layer_parameterisation_is_explicit_sharing() -> None:
    lattice = kagome_periodic(2, 2)
    ansatz = GraphHVAnsatz(lattice, 2, parameterization="per_layer", include_h0=True)
    assert ansatz.num_parameters == 2 * (ansatz.schedule.num_layers + 1)


@pytest.mark.parametrize("name", sorted(LATTICES))
def test_each_bond_gets_exactly_one_gate_per_cycle(name: str) -> None:
    lattice = LATTICES[name]
    reps = 2
    ansatz = GraphHVAnsatz(lattice, reps, include_h0=True)
    q = lattice.qubit_of_site
    expected = {frozenset((q[a], q[b])) for a, b, _ in lattice.edges}
    expected_sorted = sorted(tuple(sorted(b)) for b in expected)
    dimer_qubits = {frozenset(d) for d in ansatz.qubit_dimers}
    ops = logical_ops(ansatz.build())
    exchanges = [op for op in ops if op.kind == "exchange"]
    two = [op for op in ops if op.kind == "two"]
    assert {frozenset(op.qubits) for op in two} <= dimer_qubits  # singlet CNOTs only
    for cycle in range(1, reps + 1):
        bond_gates = [g for g in ansatz.gates if g.cycle == cycle and g.kind == "bond"]
        assert sorted(tuple(sorted(g.qubits)) for g in bond_gates) == expected_sorted
        h0 = [g for g in ansatz.gates if g.cycle == cycle and g.kind == "h0"]
        assert {frozenset(g.qubits) for g in h0} == dimer_qubits
    # every two-qubit interaction of the circuit is a lattice bond: no unintended couplings
    assert {frozenset(op.qubits) for op in exchanges} <= expected


def test_gate_order_follows_schedule_and_h0_position() -> None:
    lattice = kagome_bm_rectangle(2, 4)
    schedule = prb105_schedule(lattice)
    for position in ("start", "end"):
        ansatz = GraphHVAnsatz(lattice, 2, schedule=schedule, include_h0=True, h0_position=position)
        per_cycle = [g for g in ansatz.gates if g.cycle == 2]
        kinds = [g.kind for g in per_cycle]
        n_h0 = len(ansatz.dimers)
        if position == "start":
            assert kinds[:n_h0] == ["h0"] * n_h0
        else:
            assert kinds[-n_h0:] == ["h0"] * n_h0
        bonds = [g.sites for g in per_cycle if g.kind == "bond"]
        assert tuple(bonds) == schedule.bonds
        names = [g.parameter for g in ansatz.gates]
        assert names == sorted(names)  # alphabetical == chronological


def test_parameters_are_independent_per_gate() -> None:
    ansatz = GraphHVAnsatz(kagome_unit_cell(), 2, include_h0=True)
    bond_params = [g.parameter for g in ansatz.gates if g.kind == "bond"]
    assert len(set(bond_params)) == len(bond_params)
    h0_params = {g.parameter for g in ansatz.gates if g.kind == "h0"}
    assert len(h0_params) == 2  # one per cycle
    assert not h0_params & set(bond_params)


def test_heisenberg_gate_matches_existing_hva_convention() -> None:
    """graph_hva and the ring HVA use the same exchange gate exp(-i theta/2 (XX+YY+ZZ))."""
    lattice = custom_lattice(2, [[0, 1]])
    graph = GraphHVAnsatz(lattice, 1, dimers=[(0, 1)]).build()
    ring = HVAnsatz(2, 1).build()
    theta = 0.37
    a = Statevector(graph.assign_parameters([theta]))
    b = Statevector(ring.assign_parameters([theta, theta]))
    # on the singlet the gate only adds exp(3 i theta / 2) (and twice for the ring's 2 gates)
    assert abs(a.inner(b)) == pytest.approx(1.0)


# -- physics --------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(LATTICES))
def test_reference_state_is_singlet_product(name: str) -> None:
    lattice = LATTICES[name]
    ansatz = GraphHVAnsatz(lattice, 1, include_h0=True)
    circuit = ansatz.build()
    state = Statevector(circuit.assign_parameters(np.zeros(circuit.num_parameters)))
    energy = state.expectation_value(heisenberg_hamiltonian(lattice)).real
    assert energy == pytest.approx(-1.5 * lattice.num_sites)  # each singlet -3, rest 0
    # it is the ground state of H0 = sum over dimers
    h0 = SparsePauliOp.from_sparse_list(
        [(p, list(d), 1.0) for d in ansatz.qubit_dimers for p in ("XX", "YY", "ZZ")],
        num_qubits=lattice.num_sites,
    )
    assert state.expectation_value(h0).real == pytest.approx(-3 * len(ansatz.dimers))


@pytest.mark.parametrize("name", ["star", "bm2x4", "torus2x2"])
def test_total_spin_singlet_for_random_parameters(name: str, rng: np.random.Generator) -> None:
    lattice = LATTICES[name]
    circuit = GraphHVAnsatz(lattice, 2, include_h0=True).build()
    state = Statevector(
        circuit.assign_parameters(rng.uniform(-np.pi, np.pi, circuit.num_parameters))
    )
    s2 = state.expectation_value(_total_spin_squared(lattice.num_sites)).real
    assert s2 == pytest.approx(0.0, abs=1e-9)
    sz = SparsePauliOp.from_sparse_list(
        [("Z", [i], 0.5) for i in range(lattice.num_sites)], num_qubits=lattice.num_sites
    )
    assert state.expectation_value(sz).real == pytest.approx(0.0, abs=1e-9)


def test_phase_only_gates_and_pruning_change_nothing(rng: np.random.Generator) -> None:
    lattice = kagome_bm_rectangle(2, 6)
    schedule = prb105_schedule(lattice, order="fig3")  # H1 = dimers comes first
    full = GraphHVAnsatz(lattice, 2, schedule=schedule, include_h0=True, h0_position="start")
    pruned = GraphHVAnsatz(
        lattice, 2, schedule=schedule, include_h0=True, h0_position="start", prune_phase_gates=True
    )
    assert full.phase_only_gates == 2 * len(full.dimers)  # cycle-1 H0 block + cycle-1 H1
    assert full.num_parameters == full.num_nominal_parameters
    assert pruned.num_parameters == full.num_nominal_parameters - 1 - len(full.dimers)
    values = dict(zip(full.parameter_names, rng.uniform(-1, 1, full.num_parameters), strict=True))
    a = Statevector(full.build().assign_parameters(values))
    b = Statevector(
        pruned.build().assign_parameters({k: values[k] for k in pruned.parameter_names})
    )
    assert abs(a.inner(b)) == pytest.approx(1.0)


def test_redundant_h0_is_reported() -> None:
    lattice = kagome_bm_rectangle(2, 6)
    fig3 = GraphHVAnsatz(
        lattice, 3, schedule=prb105_schedule(lattice, order="fig3"), include_h0=True
    )
    eq4 = GraphHVAnsatz(lattice, 3, schedule=prb105_schedule(lattice), include_h0=True)
    assert len(fig3.redundant_parameters) == 2  # H0 at the end fuses with the next H1
    assert eq4.redundant_parameters == ()
    assert eq4.phase_only_gates == 0


def test_greedy_order_avoids_phase_only_gates() -> None:
    for lattice in (kagome_periodic(2, 2), kagome_unit_cell(), kagome_bm_rectangle(2, 6)):
        assert GraphHVAnsatz(lattice, 2).phase_only_gates == 0
        assert GraphHVAnsatz(lattice, 2, include_h0=True).redundant_parameters == ()


def test_unit_cell_reaches_exact_ground_state_at_zero() -> None:
    """With the spoke covering the reference state is already exact (-18)."""
    circuit = GraphHVAnsatz(kagome_unit_cell(), 1).build()
    state = Statevector(circuit.assign_parameters(np.zeros(circuit.num_parameters)))
    energy = state.expectation_value(heisenberg_hamiltonian(kagome_unit_cell())).real
    assert energy == pytest.approx(-18.0)


def test_parameter_shift_gradient_is_exact(rng: np.random.Generator) -> None:
    """Unsharing splits each exchange gate into three shiftable rotations; sum = exact grad."""
    lattice = kagome_bm_rectangle(2, 4)
    circuit = GraphHVAnsatz(lattice, 1, include_h0=True).build()
    hamiltonian = heisenberg_hamiltonian(lattice)
    unshared, pmap = unshare_parameters(circuit)
    theta = rng.uniform(-1, 1, circuit.num_parameters)

    def energy(c: QuantumCircuit, x: np.ndarray) -> float:
        return float(Statevector(c.assign_parameters(x)).expectation_value(hamiltonian).real)

    phi = pmap.expand(theta)
    shift = np.zeros_like(phi)
    grad_phi = np.empty_like(phi)
    for k in range(phi.size):
        shift[:] = 0
        shift[k] = np.pi / 2
        grad_phi[k] = (energy(unshared, phi + shift) - energy(unshared, phi - shift)) / 2
    grad = pmap.contract(grad_phi)
    eps = 1e-6
    fd = np.array(
        [
            (energy(circuit, theta + eps * e) - energy(circuit, theta - eps * e)) / (2 * eps)
            for e in np.eye(theta.size)
        ]
    )
    np.testing.assert_allclose(grad, fd, atol=1e-6)


def test_invalid_options() -> None:
    lattice = kagome_unit_cell()
    with pytest.raises(ValueError, match="reps"):
        GraphHVAnsatz(lattice, 0)
    with pytest.raises(ValueError, match="parameterization"):
        GraphHVAnsatz(lattice, 1, parameterization="shared")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not a bond"):
        GraphHVAnsatz(lattice, 1, dimers=[(0, 3), (1, 2), (4, 5), (6, 7), (8, 9), (10, 11)])
    bad = ExchangeSchedule(layers=(((0, 1),),), kind="explicit")
    with pytest.raises(ValueError, match="misses"):
        GraphHVAnsatz(lattice, 1, schedule=bad)


def test_make_schedule_dispatch() -> None:
    lattice = kagome_bm_rectangle(2, 4)
    assert make_schedule(lattice, "prb105_fig3").kind == "prb105_fig3"
    assert make_schedule(lattice, "prb106").kind == "prb106"
    with pytest.raises(ValueError, match="needs the layers"):
        make_schedule(lattice, "explicit")
