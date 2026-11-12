"""Local Pauli noise at the noise locations of PRB 106 ("paper-layer" noise).

Kattemölle and van Wezel, PRB 106, 214429 (2022), Sec. II D, study two single-qubit channels,

* depolarising  ``rho -> (1 - p) rho + p/3 (X rho X + Y rho Y + Z rho Z)``   (their Eq. 7)
* bit flip      ``rho -> (1 - p) rho + p X rho X``                            (their Eq. 8)

applied *to every qubit* after the initial-state (singlet) preparation and after every layer of
HEIS (exchange) gates: ``n d`` space-time locations with ``d = |c| p + 1``. This module
reproduces exactly that convention on the *logical* circuit of :class:`GraphHVAnsatz`. It
deliberately knows nothing about hardware: no routing, no native gates, no SWAPs, no readout
error. Noise on a compiled hardware circuit is the separate ``aer_noisy`` backend (a fake
backend's noise model on the transpiled circuit), see ``docs/experiments/graph_hva_noise.md``.

Note the parametrisation: ``p`` here is the paper's error probability ``p_e``. Qiskit's
``depolarizing_error(lam)`` is ``(1 - lam) rho + lam I/2``, which is the same channel with
``lam = 4 p / 3``.

Two simulation engines share one circuit (:func:`layer_noise_circuit`):

* :class:`PauliTrajectorySimulator` samples Pauli errors per shot (the stochastic reading of the
  channels, as in PRB 106 Sec. II E). Shots without an error share the noise-free state, so
  only erroneous shots are simulated; this is what makes noisy *optimisation* affordable.
* :func:`exact_noisy_state` runs Aer's density-matrix simulator on the same circuit, giving the
  exact mixed state for fixed-parameter analysis and for validating the sampler.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray
from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from kagome_vqe.ansatz.graph_hva import NOISE_MARK

if TYPE_CHECKING:
    from scipy.sparse import spmatrix

    from kagome_vqe.ansatz.graph_hva import GraphHVAnsatz

__all__ = [
    "NoiseChannel",
    "PauliChannel",
    "PauliTrajectorySimulator",
    "TrajectorySample",
    "aer_circuit",
    "exact_noisy_state",
    "insert_layer_noise",
    "kraus_operators",
    "layer_noise_circuit",
    "noise_locations",
    "pauli_probabilities",
    "survival_probability",
]

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

NoiseChannel = Literal["depolarizing", "bit_flip"]

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_probabilities(channel: NoiseChannel, p: float) -> tuple[float, float, float]:
    """``(p_X, p_Y, p_Z)`` of the channel with error probability ``p`` (PRB 106 Eqs. 7-8)."""
    if not 0.0 <= p <= 1.0:
        msg = f"error probability must lie in [0, 1], got {p}"
        raise ValueError(msg)
    if str(channel) not in ("depolarizing", "bit_flip"):
        msg = f"unknown channel {channel!r}"
        raise ValueError(msg)
    if channel == "depolarizing":
        return (p / 3, p / 3, p / 3)
    return (p, 0.0, 0.0)  # bit flip


def kraus_operators(channel: NoiseChannel, p: float) -> list[ComplexArray]:
    """Kraus operators ``sqrt(p_P) P`` (identity first) of the single-qubit channel."""
    px, py, pz = pauli_probabilities(channel, p)
    weights = {"I": 1.0 - px - py - pz, "X": px, "Y": py, "Z": pz}
    return [
        np.asarray(math.sqrt(w) * _PAULI[k], dtype=complex)
        for k, w in weights.items()
        if w > 0 or k == "I"
    ]


def survival_probability(channel: NoiseChannel, p: float, locations: int) -> float:
    """Probability that no error occurs at any of ``locations``: ``(1 - p)^(n d)``.

    PRB 106 Eq. 9 approximates the noisy ground-state fidelity by this factor times the
    noise-free fidelity, assuming any error makes the state orthogonal to the ground state.
    """
    del channel  # both channels err with total probability p per location
    return float((1.0 - p) ** locations)


class PauliChannel(Instruction):  # type: ignore[misc]
    """A single-qubit Pauli channel ``(1 - sum) rho + p_X X rho X + p_Y Y rho Y + p_Z Z rho Z``.

    A plain circuit instruction carrying the three probabilities, so a noisy circuit can be
    inspected, counted and serialised without Aer. :meth:`to_aer` converts it for Aer.
    """

    def __init__(self, px: float, py: float, pz: float, label: str | None = None) -> None:
        """See class docstring."""
        if min(px, py, pz) < 0 or px + py + pz > 1 + 1e-12:
            msg = f"invalid Pauli probabilities {(px, py, pz)}"
            raise ValueError(msg)
        super().__init__("pauli_channel", 1, 0, [float(px), float(py), float(pz)], label=label)

    @property
    def probabilities(self) -> tuple[float, float, float]:
        """``(p_X, p_Y, p_Z)``."""
        px, py, pz = (float(v) for v in self.params)
        return px, py, pz

    def to_aer(self) -> Any:
        """The same channel as an Aer ``QuantumError`` instruction."""
        from qiskit_aer.noise import pauli_error

        px, py, pz = self.probabilities
        terms = [("I", 1.0 - px - py - pz), ("X", px), ("Y", py), ("Z", pz)]
        return pauli_error([(k, w) for k, w in terms if w > 0]).to_instruction()


def insert_layer_noise(marked: QuantumCircuit, channel: NoiseChannel, p: float) -> QuantumCircuit:
    """Replace every barrier labelled :data:`NOISE_MARK` by the channel on *every* qubit.

    The result has the same gates in the same order; nothing is routed or re-synthesised.
    Channels with ``p = 0`` are still inserted (as identities) so that location counts stay
    visible in the circuit statistics.
    """
    probabilities = pauli_probabilities(channel, p)
    out = QuantumCircuit(*marked.qregs, *marked.cregs, name=marked.name)
    for inst in marked.data:
        op = inst.operation
        if op.name == "barrier" and op.label == NOISE_MARK:
            for qubit in out.qubits:
                out.append(PauliChannel(*probabilities, label=channel), [qubit])
        else:
            out.append(op, inst.qubits, inst.clbits)
    return out


def layer_noise_circuit(ansatz: GraphHVAnsatz, channel: NoiseChannel, p: float) -> QuantumCircuit:
    """The logical ``graph_hva`` circuit with PRB 106 layer noise."""
    return insert_layer_noise(ansatz.build(mark_noise_locations=True), channel, p)


def noise_locations(circuit: QuantumCircuit) -> int:
    """Number of single-qubit noise channels in ``circuit`` (``n d`` for layer noise)."""
    return sum(1 for inst in circuit.data if inst.operation.name == "pauli_channel")


def aer_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    """``circuit`` with every :class:`PauliChannel` replaced by its Aer equivalent."""
    out = QuantumCircuit(*circuit.qregs, *circuit.cregs, name=circuit.name)
    for inst in circuit.data:
        op = inst.operation
        out.append(op.to_aer() if isinstance(op, PauliChannel) else op, inst.qubits, inst.clbits)
    return out


def exact_noisy_state(circuit: QuantumCircuit, theta: FloatArray) -> ComplexArray:
    """Exact density matrix of the noisy circuit at ``theta`` (Aer, ``density_matrix``).

    Memory is ``16 * 4**n`` bytes: 268 MB at 12 qubits; do not use beyond that.
    """
    from qiskit_aer import AerSimulator

    if circuit.num_qubits > 12:
        msg = f"exact density matrices are limited to 12 qubits, got {circuit.num_qubits}"
        raise ValueError(msg)
    bound = aer_circuit(circuit.assign_parameters(np.asarray(theta, dtype=float)))
    bound.save_density_matrix()
    result = AerSimulator(method="density_matrix").run(bound).result()
    return np.asarray(result.data()["density_matrix"], dtype=complex)


# -- trajectory simulation ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Op:
    """A fused unitary on one or two qubits (tensor axes already resolved)."""

    qubits: tuple[int, ...]
    matrix: ComplexArray


@dataclass(frozen=True, slots=True)
class _Segment:
    """Gates up to the next noise location, then the channels at that location."""

    gates: tuple[Instruction, ...]
    gate_qubits: tuple[tuple[int, ...], ...]
    channel_qubits: tuple[int, ...]
    channel_probabilities: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True, slots=True)
class TrajectorySample:
    """Per-trajectory expectation values at fixed parameters (the PRB 106 estimator).

    The distinct values are stored once with how often they occurred, so samples with millions
    of trajectories (needed at small ``p``) stay small: all error-free trajectories share the
    noise-free values.

    Attributes:
        values: ``(m, k)`` distinct per-trajectory values of ``k`` observables.
        counts: ``(m,)`` multiplicities; ``counts.sum()`` is the number of trajectories.
        error_free: Trajectories without any error (``counts[0]`` when positive).
    """

    values: FloatArray
    counts: NDArray[np.int64]
    error_free: int

    @property
    def trajectories(self) -> int:
        """Total number of sampled trajectories."""
        return int(self.counts.sum())

    def mean(self) -> FloatArray:
        """Sample mean of each observable."""
        return np.asarray(self.counts @ self.values / self.trajectories, dtype=float)

    def bootstrap_ci(
        self, rng: np.random.Generator, *, resamples: int = 2000, level: float = 0.95
    ) -> FloatArray:
        """Percentile bootstrap confidence interval of each mean, shape ``(k, 2)``.

        Resampling ``N`` trajectories with replacement is a multinomial draw over the distinct
        values with the observed frequencies.
        """
        weights = self.counts / self.trajectories
        draws = rng.multinomial(self.trajectories, weights, size=resamples)
        means = draws @ self.values / self.trajectories
        tail = (1 - level) / 2
        return np.asarray(np.quantile(means, [tail, 1 - tail], axis=0).T, dtype=float)


class PauliTrajectorySimulator:
    """Monte Carlo trajectories of a circuit whose only non-unitary parts are Pauli channels.

    The circuit (from :func:`layer_noise_circuit`) is split into unitary segments separated by
    noise locations. Per call the noise-free state is propagated once and cached after every
    segment; a trajectory whose first error occurs at location ``l`` starts from that cached
    state, so only erroneous trajectories cost a simulation, and they are propagated together
    as one batch.

    Args:
        circuit: Parametrised circuit of 1- and 2-qubit gates and :class:`PauliChannel` s.
        max_batch_amplitudes: Upper bound on ``trajectories * 2**n`` held at once.
    """

    def __init__(self, circuit: QuantumCircuit, *, max_batch_amplitudes: int = 2**25) -> None:
        """See class docstring."""
        self._circuit = circuit
        self._n = circuit.num_qubits
        self._dim = 2**self._n
        self._max_batch = max(1, max_batch_amplitudes // self._dim)
        self._parameters = list(circuit.parameters)
        self._segments = self._split(circuit)
        locations = [
            (s, q, probs)
            for s, seg in enumerate(self._segments)
            for q, probs in zip(seg.channel_qubits, seg.channel_probabilities, strict=True)
        ]
        self._loc_segment = np.array([s for s, _, _ in locations], dtype=np.int64)
        self._loc_qubit = np.array([q for _, q, _ in locations], dtype=np.int64)
        probs = np.array([p for _, _, p in locations], dtype=float).reshape(-1, 3)
        self._loc_total = probs.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            self._loc_cumulative = np.where(
                self._loc_total[:, None] > 0,
                np.cumsum(probs, axis=1) / self._loc_total[:, None],
                1.0,
            )

    @property
    def num_qubits(self) -> int:
        """Register width."""
        return self._n

    @property
    def num_locations(self) -> int:
        """Single-qubit noise locations (``n d`` for layer noise)."""
        return int(self._loc_qubit.size)

    @property
    def num_parameters(self) -> int:
        """Circuit parameters, in ``circuit.parameters`` order."""
        return len(self._parameters)

    @property
    def error_free_probability(self) -> float:
        """Probability that a trajectory has no error at all."""
        return float(np.prod(1.0 - self._loc_total))

    # -- circuit parsing ------------------------------------------------------------------------

    def _split(self, circuit: QuantumCircuit) -> list[_Segment]:
        segments: list[_Segment] = []
        gates: list[Instruction] = []
        gate_qubits: list[tuple[int, ...]] = []
        ch_qubits: list[int] = []
        ch_probs: list[tuple[float, float, float]] = []

        def flush() -> None:
            segments.append(
                _Segment(tuple(gates), tuple(gate_qubits), tuple(ch_qubits), tuple(ch_probs))
            )
            gates.clear()
            gate_qubits.clear()
            ch_qubits.clear()
            ch_probs.clear()

        for inst in circuit.data:
            op = inst.operation
            qubits = tuple(circuit.find_bit(q).index for q in inst.qubits)
            if isinstance(op, PauliChannel):
                ch_qubits.append(qubits[0])
                ch_probs.append(op.probabilities)
                continue
            if op.name == "barrier":
                continue
            if op.name in ("measure", "reset") or op.num_qubits > 2:
                msg = f"trajectory simulation supports 1- and 2-qubit gates, got {op.name!r}"
                raise ValueError(msg)
            if ch_qubits:  # a gate after channels closes the previous segment
                flush()
            gates.append(op)
            gate_qubits.append(qubits)
        flush()
        return segments

    def _bind(self, theta: FloatArray) -> list[list[_Op]]:
        """Fused numeric unitaries per segment."""
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.size != len(self._parameters):
            msg = f"expected {len(self._parameters)} parameters, got {theta.size}"
            raise ValueError(msg)
        values = dict(zip(self._parameters, theta, strict=True))
        bound: list[list[_Op]] = []
        for seg in self._segments:
            ops: list[_Op] = []
            for op, qubits in zip(seg.gates, seg.gate_qubits, strict=True):
                gate = op
                if op.params and any(hasattr(v, "parameters") for v in op.params):
                    gate = op.copy()
                    gate.params = [
                        float(v.bind({k: values[k] for k in v.parameters}))
                        if hasattr(v, "parameters")
                        else v
                        for v in op.params
                    ]
                ops = _fuse(ops, qubits, np.asarray(gate.to_matrix(), dtype=complex))
            bound.append(ops)
        return bound

    # -- state evolution ------------------------------------------------------------------------

    def _apply(self, states: ComplexArray, op: _Op) -> ComplexArray:
        """Apply a 1- or 2-qubit unitary to a batch of states, shape ``(K, 2**n)``."""
        return _apply_unitary(states, op.matrix, op.qubits, self._n)

    def _segment_states(self, ops: list[list[_Op]]) -> list[ComplexArray]:
        """Noise-free state after the gates of each segment (before its channels)."""
        state = np.zeros((1, self._dim), dtype=complex)
        state[0, 0] = 1.0
        cache = []
        for seg_ops in ops:
            for op in seg_ops:
                state = self._apply(state, op)
            cache.append(state[0].copy())
        return cache

    def _sample_errors(
        self, rng: np.random.Generator, trajectories: int
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
        """Errors of ``trajectories`` shots: (trajectory, location, pauli 1..3) triples."""
        if trajectories == 0 or self.num_locations == 0:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty, empty
        # Number of errors per location is binomial; positions are then uniform.
        per_location = rng.binomial(trajectories, self._loc_total)
        loc = np.repeat(np.arange(self.num_locations), per_location)
        traj = np.concatenate(
            [rng.choice(trajectories, size=k, replace=False) for k in per_location if k > 0]
            or [np.zeros(0, dtype=np.int64)]
        ).astype(np.int64)
        u = rng.random(loc.size)
        pauli = 1 + (u[:, None] >= self._loc_cumulative[loc][:, :2]).sum(axis=1)
        return traj, loc, pauli.astype(np.int64)

    def _erroneous_states(
        self,
        ops: list[list[_Op]],
        cache: list[ComplexArray],
        traj: NDArray[np.int64],
        loc: NDArray[np.int64],
        pauli: NDArray[np.int64],
    ) -> tuple[NDArray[np.int64], ComplexArray]:
        """Final states of the trajectories with at least one error.

        Returns:
            ``(ids, states)``: the trajectory indices (sorted) and their final states.
        """
        ids = np.unique(traj)
        if ids.size == 0:
            return ids, np.zeros((0, self._dim), dtype=complex)
        out = np.empty((ids.size, self._dim), dtype=complex)
        for start in range(0, ids.size, self._max_batch):
            chunk = ids[start : start + self._max_batch]
            mask = np.isin(traj, chunk)
            out[start : start + chunk.size] = self._propagate(
                ops,
                cache,
                np.searchsorted(chunk, traj[mask]),
                loc[mask],
                pauli[mask],
                chunk.size,
            )
        return ids, out

    def _propagate(
        self,
        ops: list[list[_Op]],
        cache: list[ComplexArray],
        rows: NDArray[np.int64],
        loc: NDArray[np.int64],
        pauli: NDArray[np.int64],
        size: int,
    ) -> ComplexArray:
        seg_of_error = self._loc_segment[loc]
        first = np.full(size, len(self._segments), dtype=np.int64)
        np.minimum.at(first, rows, seg_of_error)
        order = np.argsort(first, kind="stable")  # batch row b holds trajectory order[b]
        position = np.empty(size, dtype=np.int64)
        position[order] = np.arange(size)
        first_sorted = first[order]
        states = np.zeros((0, self._dim), dtype=complex)
        for s in range(len(self._segments)):
            if s > 0 and states.shape[0]:
                for op in ops[s]:
                    states = self._apply(states, op)
            joining = int(np.searchsorted(first_sorted, s, side="right")) - states.shape[0]
            if joining > 0:
                states = np.vstack([states, np.repeat(cache[s][None, :], joining, axis=0)])
            here = seg_of_error == s
            if not here.any():
                continue
            for q in np.unique(self._loc_qubit[loc[here]]):
                for kind in (1, 2, 3):
                    sel = here & (self._loc_qubit[loc] == q) & (pauli == kind)
                    if not sel.any():
                        continue
                    # An even number of identical Paulis on one row cancels (rare at small p).
                    hit, times = np.unique(position[rows[sel]], return_counts=True)
                    hit = hit[times % 2 == 1]
                    if hit.size:
                        states[hit] = _apply_unitary(
                            states[hit], _PAULI["IXYZ"[kind]], (int(q),), self._n
                        )
        return states

    # -- public API -----------------------------------------------------------------------------

    def ideal_state(self, theta: FloatArray) -> ComplexArray:
        """The noise-free final state (qiskit little-endian order)."""
        return self._segment_states(self._bind(theta))[-1]

    def sample_measurements(
        self,
        theta: FloatArray,
        bases: Sequence[str],
        shots: int,
        rng: np.random.Generator,
    ) -> list[NDArray[np.int64]]:
        """Measurement outcome histograms, one per basis, ``shots`` trajectories each.

        Every shot is an independent trajectory (its own Pauli errors) followed by one
        projective measurement, exactly as one execution of the circuit on hardware.

        Args:
            theta: Circuit parameters.
            bases: Per basis a string of ``n`` letters ``X``/``Y``/``Z``, character ``q`` for
                qubit ``q``.
            shots: Shots per basis.
            rng: Random source.

        Returns:
            Per basis an array of ``2**n`` outcome counts (index bit ``q`` = qubit ``q``).
        """
        ops = self._bind(theta)
        cache = self._segment_states(ops)
        ideal = cache[-1]
        histograms = []
        for basis in bases:
            traj, loc, pauli = self._sample_errors(rng, shots)
            ids, states = self._erroneous_states(ops, cache, traj, loc, pauli)
            counts = np.zeros(self._dim, dtype=np.int64)
            clean = shots - ids.size
            if clean:
                probs = _basis_probabilities(ideal[None, :], basis, self._n)[0]
                counts += rng.multinomial(clean, probs / probs.sum())
            if ids.size:
                probs = _basis_probabilities(states, basis, self._n)
                cumulative = np.cumsum(probs, axis=1)
                u = rng.random(ids.size) * cumulative[:, -1]
                outcomes = (cumulative < u[:, None]).sum(axis=1)
                counts += np.bincount(outcomes, minlength=self._dim)
            histograms.append(counts)
        return histograms

    def sample_expectations(
        self,
        theta: FloatArray,
        operators: Sequence[spmatrix | ComplexArray],
        trajectories: int,
        rng: np.random.Generator,
    ) -> TrajectorySample:
        """Exact expectation values per trajectory (PRB 106's estimator, Sec. II E).

        Args:
            theta: Circuit parameters.
            operators: Per observable either a sparse/dense ``2**n`` square matrix (value
                ``<psi|A|psi>``) or a ``(2**n, m)`` orthonormal basis ``V`` of a subspace
                (value ``|V^dag psi|^2``, e.g. the ground-space fidelity).
            trajectories: Number of trajectories ``N``.
            rng: Random source.
        """
        ops = self._bind(theta)
        cache = self._segment_states(ops)
        traj, loc, pauli = self._sample_errors(rng, trajectories)
        ids, states = self._erroneous_states(ops, cache, traj, loc, pauli)
        all_states = np.vstack([cache[-1][None, :], states])
        values = np.column_stack([_expectations(all_states, op) for op in operators])
        clean = trajectories - ids.size
        counts = np.concatenate([[clean], np.ones(ids.size, dtype=np.int64)]).astype(np.int64)
        return TrajectorySample(values=values, counts=counts, error_free=int(clean))


# -- helpers --------------------------------------------------------------------------------------


def _fuse(ops: list[_Op], qubits: tuple[int, ...], matrix: ComplexArray) -> list[_Op]:
    """Append a gate, multiplying it into the previous op when it acts within its qubits."""
    if ops and set(qubits) <= set(ops[-1].qubits):
        last = ops[-1]
        embedded = _embed(matrix, qubits, last.qubits)
        ops[-1] = _Op(last.qubits, embedded @ last.matrix)
        return ops
    ops.append(_Op(qubits, matrix))
    return ops


def _embed(matrix: ComplexArray, qubits: tuple[int, ...], target: tuple[int, ...]) -> ComplexArray:
    """Express a gate on ``qubits`` as a matrix on ``target`` (little-endian, qiskit order)."""
    if qubits == target:
        return matrix
    if len(target) == 2 and len(qubits) == 2:
        swap = np.eye(4, dtype=complex)[[0, 2, 1, 3]]
        return np.asarray(swap @ matrix @ swap, dtype=complex)
    if len(target) == 2 and len(qubits) == 1:
        eye = np.eye(2, dtype=complex)
        # target[0] is the low bit
        return np.asarray(
            np.kron(eye, matrix) if qubits[0] == target[0] else np.kron(matrix, eye), dtype=complex
        )
    msg = f"cannot embed a gate on {qubits} into {target}"
    raise ValueError(msg)


def _apply_unitary(
    states: ComplexArray, matrix: ComplexArray, qubits: tuple[int, ...], n: int
) -> ComplexArray:
    """``(K, 2**n)`` states, ``matrix`` on ``qubits`` (qiskit order: ``qubits[0]`` low bit)."""
    k = states.shape[0]
    if len(qubits) == 1:
        (q,) = qubits
        view = states.reshape(k * 2 ** (n - 1 - q), 2, 2**q)
        return np.asarray(np.matmul(matrix, view).reshape(k, -1), dtype=complex)
    lo, hi = qubits
    tensor = matrix.reshape(2, 2, 2, 2)  # [out_hi, out_lo, in_hi, in_lo] for (qubits[1], [0])
    if lo > hi:
        lo, hi = hi, lo
        tensor = tensor.transpose(1, 0, 3, 2)
    view = states.reshape(k * 2 ** (n - 1 - hi), 2, 2 ** (hi - lo - 1), 2, 2**lo)
    out = np.einsum("ABab,LaMbR->LAMBR", tensor, view, optimize=True)
    return np.ascontiguousarray(out).reshape(k, -1)


_BASIS_CHANGE = {
    "Z": np.eye(2, dtype=complex),
    "X": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),  # H
    "Y": np.array([[1, -1j], [1, 1j]], dtype=complex) / np.sqrt(2),  # H Sdg
}


def _basis_probabilities(states: ComplexArray, basis: str, n: int) -> FloatArray:
    """Outcome probabilities after rotating each qubit into ``basis[q]``."""
    rotated = states
    for q, letter in enumerate(basis):
        if letter != "Z":
            rotated = _apply_unitary(rotated, _BASIS_CHANGE[letter], (q,), n)
    return np.asarray(np.abs(rotated) ** 2, dtype=float)


def _expectations(states: ComplexArray, operator: spmatrix | ComplexArray) -> FloatArray:
    """``<psi|A|psi>`` per row, or ``|V^dag psi|^2`` for a ``(2**n, m)`` basis ``V``."""
    dim = states.shape[1]
    if operator.shape == (dim, dim):
        applied = (operator @ states.T).T
        return np.asarray(np.real(np.sum(states.conj() * applied, axis=1)), dtype=float)
    overlaps = states.conj() @ np.asarray(operator)
    return np.asarray(np.sum(np.abs(overlaps) ** 2, axis=1), dtype=float)


def measurement_bases(observable: SparsePauliOp) -> list[tuple[str, FloatArray]]:
    """Qubit-wise commuting groups as ``(basis, diagonal)`` pairs.

    ``basis[q]`` is the letter measured on qubit ``q`` (``Z`` where the group acts trivially),
    and ``diagonal[x]`` is the group's eigenvalue on outcome ``x`` in that basis, so the group's
    expectation from a histogram ``c`` of ``N`` shots is ``c @ diagonal / N``.
    """
    n = observable.num_qubits
    outcomes = np.arange(2**n, dtype=np.int64)
    result = []
    for group in observable.group_commuting(qubit_wise=True):
        letters = ["Z"] * n
        diagonal = np.zeros(2**n, dtype=float)
        for label, coeff in group.to_list():
            mask = 0
            for q in range(n):
                letter = label[n - 1 - q]
                if letter != "I":
                    letters[q] = letter
                    mask |= 1 << q
            parity = (np.bitwise_count(outcomes & mask) & 1).astype(np.int64)
            diagonal += float(np.real(coeff)) * (1 - 2 * parity)
        result.append(("".join(letters), diagonal))
    return result
