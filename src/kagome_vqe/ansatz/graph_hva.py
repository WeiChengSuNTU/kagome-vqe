"""Graph-based Hamiltonian variational ansatz with one parameter per gate (``graph_hva``).

The circuit is built from the *interaction graph* of a :class:`~kagome_vqe.lattice.Lattice`
alone; no ring, no fixed site count and no hardware enter here.

Conventions
-----------
Hamiltonian (as everywhere in this package, Pauli normalisation)::

    H = sum_<ij> J_ij (X_i X_j + Y_i Y_j + Z_i Z_j) = 4 sum_<ij> J_ij S_i . S_j

Exchange gate on bond ``(i, j)`` with angle ``theta``::

    G_ij(theta) = exp(-i theta/2 (X_i X_j + Y_i Y_j + Z_i Z_j)) = exp(-2 i theta S_i . S_j)
                = RZZ(theta) RYY(theta) RXX(theta)       (the three factors commute)

which is the same gate as in :class:`~kagome_vqe.ansatz.HVAnsatz`. It is SU(2) symmetric, so
the ansatz never leaves the total-spin sector of the reference state. The coupling ``J_ij`` is
*absorbed* into ``theta``: the ansatz uses the bond graph, not the weights. In the notation of
PRB 106, ``HEIS(alpha) = exp(-i alpha S.S)`` up to a phase, so ``alpha = 2 theta``; in PRB 105
``exp(-i t X.X)``, so ``t = theta / 2``.

Reference state
---------------
A product of singlets ``(|01> - |10>)/sqrt(2)`` on a dimer covering ``D`` (a perfect matching
of lattice bonds). It is a ground state of ``H0 = sum_{(ij) in D} (XX + YY + ZZ)_{ij}``.

Circuit (chronological)
-----------------------
::

    singlets on D
    for cycle c = 1 .. p:
        [H0 block, if include_h0 and h0_position == "start"]
        for layer in schedule.layers:          # each layer: disjoint bonds
            for bond in layer: G_bond(theta_{c, bond})
        [H0 block, if include_h0 and h0_position == "end"]

where the H0 block is ``exp(-i phi_c/2 H0) = prod_{(ij) in D} G_ij(phi_c)``: one extra parameter
per cycle, never shared with a bond gate. Parameter counts: ``p * N_b`` without H0 and
``p * (N_b + 1)`` with it (one parameter per gate, "OPG"). The ``"per_layer"``
parameterisation instead shares one parameter across each layer (PRB 105's "per edge colour"),
giving ``p * N_layers`` (+ ``p``). Sharing only ever happens when asked for.

Global-phase gates
------------------
In the first cycle, a gate on a reference dimer whose two sites have not been touched yet acts
on an exact singlet and only multiplies the state by ``exp(3 i theta / 2)``. The H0 block at
the start of cycle 1 is entirely of this kind. :attr:`GraphHVAnsatz.phase_only_gates` reports
them; with ``prune_phase_gates=True`` they are removed (and a parameter that drives only such
gates disappears), which changes no state and is reflected in :attr:`num_parameters` versus
:attr:`num_nominal_parameters`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

from qiskit.circuit import Parameter, QuantumCircuit

from kagome_vqe.ansatz.gates import singlet_pair
from kagome_vqe.ansatz.hva import heisenberg_bond
from kagome_vqe.lattice import Lattice, find_dimer_covering
from kagome_vqe.schedule import ExchangeSchedule, greedy_schedule

__all__ = ["NOISE_MARK", "GateSpec", "GraphHVAnsatz", "H0Position", "Parameterization"]

NOISE_MARK = "noise_location"
"""Label of the barriers :meth:`GraphHVAnsatz.build` places at the PRB 106 noise locations."""

Parameterization = Literal["opg", "per_layer"]
H0Position = Literal["start", "end"]


@dataclass(frozen=True, slots=True)
class GateSpec:
    """One exchange gate of the logical circuit, in chronological order.

    Attributes:
        cycle: 1-based HVA cycle.
        kind: ``"bond"`` for a Hamiltonian bond, ``"h0"`` for a gate of the H0 block.
        sites: The bond in site labels.
        qubits: The bond in ansatz-qubit labels.
        layer: Schedule layer index (``-1`` for H0 gates).
        parameter: Name of the circuit parameter driving the gate.
        phase_only: Whether the gate only contributes a global phase (see module docs).
        pruned: Whether the gate was left out of the circuit.
    """

    cycle: int
    kind: Literal["bond", "h0"]
    sites: tuple[int, int]
    qubits: tuple[int, int]
    layer: int
    parameter: str
    phase_only: bool
    pruned: bool


class GraphHVAnsatz:
    """Hamiltonian variational ansatz on an arbitrary interaction graph.

    Args:
        lattice: The interaction graph (bonds in site labels, ``site_order`` respected).
        reps: Number of HVA cycles ``p``.
        schedule: Gate order of one cycle; defaults to :func:`~kagome_vqe.schedule.
            greedy_schedule`.
        dimers: Reference dimer covering in site labels; defaults to ``lattice.dimers``, else
            the lexicographically first perfect matching.
        include_h0: Add an independently parameterised H0 evolution to every cycle.
        h0_position: Where the H0 block sits in each cycle.
        parameterization: ``"opg"`` (default) or ``"per_layer"``.
        prune_phase_gates: Drop the gates that only contribute a global phase.

    Raises:
        ValueError: If the graph has no dimer covering, the covering or schedule is invalid, or
            an option is unknown.
    """

    def __init__(
        self,
        lattice: Lattice,
        reps: int = 1,
        *,
        schedule: ExchangeSchedule | None = None,
        dimers: Sequence[tuple[int, int]] | None = None,
        include_h0: bool = False,
        h0_position: H0Position = "end",
        parameterization: Parameterization = "opg",
        prune_phase_gates: bool = False,
    ) -> None:
        """See class docstring."""
        if reps < 1:
            msg = f"reps must be >= 1, got {reps}"
            raise ValueError(msg)
        if parameterization not in ("opg", "per_layer"):
            msg = f"unknown parameterization {parameterization!r}"
            raise ValueError(msg)
        if h0_position not in ("start", "end"):
            msg = f"h0_position must be 'start' or 'end', got {h0_position!r}"
            raise ValueError(msg)
        if lattice.num_bonds == 0:
            msg = f"{lattice.name} has no bonds"
            raise ValueError(msg)
        if dimers is None:
            dimers = lattice.dimers if lattice.dimers is not None else find_dimer_covering(lattice)
        checked = lattice.with_dimers(dimers)  # validates the covering against the bonds
        assert checked.dimers is not None
        self._lattice = lattice
        self._reps = reps
        self._dimers: tuple[tuple[int, int], ...] = checked.dimers
        self._schedule = schedule or greedy_schedule(lattice, self._dimers, include_h0=include_h0)
        self._schedule.validate(lattice)
        self._include_h0 = include_h0
        self._h0_position: H0Position = h0_position
        self._parameterization: Parameterization = parameterization
        self._prune = prune_phase_gates
        self._gates = self._plan()

    # -- description --------------------------------------------------------------------------

    @property
    def name(self) -> str:
        """``"graph_hva"``."""
        return "graph_hva"

    @property
    def lattice(self) -> Lattice:
        """The interaction graph."""
        return self._lattice

    @property
    def num_qubits(self) -> int:
        """One qubit per site."""
        return self._lattice.num_sites

    @property
    def reps(self) -> int:
        """Number of cycles ``p``."""
        return self._reps

    @property
    def schedule(self) -> ExchangeSchedule:
        """Gate order of one cycle (site labels)."""
        return self._schedule

    @property
    def dimers(self) -> tuple[tuple[int, int], ...]:
        """Reference dimer covering in site labels."""
        return self._dimers

    @property
    def qubit_dimers(self) -> tuple[tuple[int, int], ...]:
        """Reference dimer covering in ansatz-qubit labels."""
        return self._lattice.qubit_pairs(self._dimers)

    @property
    def include_h0(self) -> bool:
        """Whether each cycle carries an H0 block."""
        return self._include_h0

    @property
    def gates(self) -> tuple[GateSpec, ...]:
        """Every exchange gate (including pruned ones) in chronological order."""
        return self._gates

    @property
    def phase_only_gates(self) -> int:
        """Number of gates that only contribute a global phase (pruned or not)."""
        return sum(g.phase_only for g in self._gates)

    @property
    def redundant_parameters(self) -> tuple[str, ...]:
        """Parameters that are pure gauge freedoms of the circuit.

        An H0 gate on dimer ``(a, b)`` *fuses* with a bond gate on the same pair when no gate
        touching ``a`` or ``b`` sits between them: ``G(phi) G(theta) = G(phi + theta)``. If every
        gate of an H0 block fuses with an independently parameterised bond gate, the block's
        parameter can be absorbed into those bonds and carries no variational freedom. This
        happens, e.g., for the PRB 105 schedule (whose first group ``H1`` is the dimer
        covering) when H0 is adjacent to ``H1``; reorder the layers to avoid it.
        """
        live = [g for g in self._gates if not g.pruned]
        fuses: dict[str, list[bool]] = {}
        for i, gate in enumerate(live):
            if gate.kind != "h0":
                continue
            pair = set(gate.qubits)
            neighbours = [
                next((h for h in reversed(live[:i]) if set(h.qubits) & pair), None),
                next((h for h in live[i + 1 :] if set(h.qubits) & pair), None),
            ]
            fused = any(
                h is not None and h.kind == "bond" and set(h.qubits) == pair for h in neighbours
            )
            fuses.setdefault(gate.parameter, []).append(fused)
        return tuple(sorted(name for name, flags in fuses.items() if all(flags)))

    @property
    def num_nominal_parameters(self) -> int:
        """``p * N_b`` (+ ``p`` with H0) for OPG; ``p * N_layers`` (+ ``p``) per layer."""
        per_cycle = (
            self._lattice.num_bonds
            if self._parameterization == "opg"
            else self._schedule.num_layers
        )
        return self._reps * (per_cycle + int(self._include_h0))

    @property
    def num_parameters(self) -> int:
        """Free parameters of the built circuit (nominal minus pruned ones)."""
        return len(self.parameter_names)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Circuit parameter names in the circuit's (alphabetical = chronological) order."""
        names = {g.parameter for g in self._gates if not g.pruned}
        return tuple(sorted(names))

    def describe(self) -> dict[str, object]:
        """Summary for dry runs and result files."""
        return {
            "name": self.name,
            "reps": self._reps,
            "parameterization": self._parameterization,
            "include_h0": self._include_h0,
            "h0_position": self._h0_position if self._include_h0 else None,
            "prune_phase_gates": self._prune,
            "schedule": self._schedule.kind,
            "schedule_layers": self._schedule.num_layers,
            "schedule_labels": list(self._schedule.labels),
            "num_bonds": self._lattice.num_bonds,
            "dimers": [list(d) for d in self._dimers],
            "nominal_parameters": self.num_nominal_parameters,
            "effective_parameters": self.num_parameters,
            "phase_only_gates": self.phase_only_gates,
            "redundant_parameters": len(self.redundant_parameters),
            "exchange_gates": sum(not g.pruned for g in self._gates),
        }

    # -- construction -------------------------------------------------------------------------

    def _plan(self) -> tuple[GateSpec, ...]:
        """Chronological gate list with parameter names and phase-only flags."""
        q = self._lattice.qubit_of_site
        dimer_keys = {frozenset(d) for d in self._dimers}
        touched: set[int] = set()
        specs: list[GateSpec] = []
        for cycle in range(1, self._reps + 1):
            slot = 0  # chronological parameter slot within the cycle

            def name(slot_index: int, cycle: int = cycle) -> str:
                return f"t{cycle:03d}_{slot_index:04d}"

            def add(
                kind: Literal["bond", "h0"],
                sites: tuple[int, int],
                layer: int,
                param: str,
                cycle: int = cycle,
            ) -> None:
                a, b = sites
                phase_only = (
                    frozenset(sites) in dimer_keys and a not in touched and b not in touched
                )
                if not phase_only:
                    touched.update(sites)
                specs.append(
                    GateSpec(
                        cycle=cycle,
                        kind=kind,
                        sites=sites,
                        qubits=(q[a], q[b]),
                        layer=layer,
                        parameter=param,
                        phase_only=phase_only,
                        pruned=False,
                    )
                )

            def h0_block(slot_index: int) -> None:
                for dimer in self._dimers:
                    add("h0", dimer, -1, name(slot_index))

            if self._include_h0 and self._h0_position == "start":
                h0_block(slot)
                slot += 1
            for layer_index, layer in enumerate(self._schedule.layers):
                if self._parameterization == "per_layer":
                    for bond in layer:
                        add("bond", bond, layer_index, name(slot))
                    slot += 1
                else:
                    for bond in layer:
                        add("bond", bond, layer_index, name(slot))
                        slot += 1
            if self._include_h0 and self._h0_position == "end":
                h0_block(slot)
                slot += 1
        if not self._prune:
            return tuple(specs)
        return tuple(replace(g, pruned=g.phase_only) for g in specs)

    @property
    def noise_layers(self) -> int:
        """Noise locations per qubit in the PRB 106 convention: ``d = |c| p + 1``.

        One after the singlet preparation and one after every layer of exchange gates; with H0
        the H0 block of each cycle is one more layer. Pruned (phase-only) gates do not remove a
        location: the layer still occupies a time step.
        """
        per_cycle = self._schedule.num_layers + (1 if self._include_h0 else 0)
        return per_cycle * self._reps + 1

    def build(self, *, mark_noise_locations: bool = False) -> QuantumCircuit:
        """Construct the logical circuit (a fresh object on every call).

        Args:
            mark_noise_locations: Insert a barrier labelled :data:`NOISE_MARK` after the
                singlet preparation and after every gate layer (:attr:`noise_layers` in
                total). :func:`kagome_vqe.noise.insert_layer_noise` replaces them with noise
                channels. Barriers change no state; without the flag the circuit is unchanged.
        """
        circuit = QuantumCircuit(self.num_qubits, name=self.name)
        for a, b in self.qubit_dimers:
            singlet_pair(circuit, a, b)
        params: dict[str, Parameter] = {}
        previous: tuple[int, str, int] | None = None
        for gate in self._gates:
            block = (gate.cycle, gate.kind, gate.layer)
            if mark_noise_locations and block != previous:
                circuit.barrier(label=NOISE_MARK)
            previous = block
            if gate.pruned:
                continue
            theta = params.setdefault(gate.parameter, Parameter(gate.parameter))
            heisenberg_bond(circuit, theta, *gate.qubits)
        if mark_noise_locations:
            circuit.barrier(label=NOISE_MARK)
        return circuit

    def __repr__(self) -> str:
        """Debug representation."""
        return (
            f"GraphHVAnsatz({self._lattice.name!r}, reps={self._reps}, "
            f"schedule={self._schedule.kind!r}, include_h0={self._include_h0}, "
            f"parameterization={self._parameterization!r})"
        )
