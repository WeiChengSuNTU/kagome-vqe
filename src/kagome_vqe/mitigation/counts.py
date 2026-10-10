"""Energies from raw measurement counts, with readout mitigation and symmetry filtering.

The Estimator primitives return expectation values only. Two mitigation techniques need the
individual shots instead, so this module measures the circuit itself:

* **Readout mitigation.** Each qubit's readout is calibrated once (prepare all-0 and all-1,
  measure) to get its flip probabilities ``e0 = P(1|0)`` and ``e1 = P(0|1)``. For independent
  readout errors, replacing each measured bit ``b`` by the weight ``v_q[b]``, where
  ``v_q = [1, -1] A_q^{-1}`` and ``A_q`` is the qubit's confusion matrix, makes the per-shot
  product over a Pauli term's support an *unbiased* estimator of its noise-free expectation.
  Without mitigation the weights are simply ``[1, -1]`` (the eigenvalues). Because the
  correction is per shot, the reported standard error is the exact one,
  ``std(per-shot energy) / sqrt(shots)``, not the primitive's bound.
* **Symmetry filtering.** The Heisenberg model conserves total spin, and the RVB and HVA states
  are total-spin singlets for every parameter value. A singlet has zero magnetisation along
  every axis, so in each measurement basis (X, Y or Z) a noise-free shot always has exactly
  ``n/2`` ones. Shots that violate this were corrupted by noise and are discarded.

When both are enabled the order matters. Filtering first and then weighting the kept shots is
biased: the filter already removes nearly every single readout flip, so the weights over-correct
each two-body term by about ``1 / ((1 - 2 e_a)(1 - 2 e_b))``, roughly 9 % on FakeGuadalupeV2.
Under full noise that over-correction happened to cancel the gate-noise residual the filter
cannot see, which made the biased combination look accurate. The combination therefore uses
*correct, then filter*: the measured distribution over all ``2**n`` outcomes is multiplied by the
inverse of the calibrated tensored readout channel, giving quasi-probabilities, which are then
restricted to the singlet sector and renormalised. This is exact for independent readout errors
and costs ``O(n 2**n)`` per basis, so it is limited to :data:`SECTOR_AWARE_MAX_QUBITS` qubits.
Its standard error comes from a bootstrap over shots.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Protocol, Self

import numpy as np
from numpy.typing import NDArray
from qiskit.circuit import ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager

from kagome_vqe.config import BackendConfig
from kagome_vqe.estimators import (
    DENSITY_MATRIX_MAX_QUBITS,
    EnergyBatch,
    RuntimeSession,
    aer_noise_model,
)
from kagome_vqe.transpile import IsaProblem

__all__ = [
    "SECTOR_AWARE_MAX_QUBITS",
    "CountsEnergy",
    "MeasurementGroup",
    "ReadoutCalibration",
    "measurement_plan",
]

log = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_SIGNS = np.array([1.0, -1.0])

SECTOR_AWARE_MAX_QUBITS = 20
"""Largest register for the distribution-level readout correction (``2**n`` quasi-probabilities)."""

_BOOTSTRAP = 50


@dataclass(frozen=True, slots=True)
class MeasurementGroup:
    """Pauli terms measurable together in one basis.

    Attributes:
        basis: Measurement basis per logical qubit, ``"X"``, ``"Y"`` or ``"Z"``.
        coeffs: Real coefficient of each term.
        supports: Logical qubits each term acts on.
    """

    basis: tuple[str, ...]
    coeffs: FloatArray
    supports: tuple[tuple[int, ...], ...]


def measurement_plan(observable: SparsePauliOp) -> tuple[list[MeasurementGroup], float]:
    """Split ``observable`` into qubit-wise commuting groups.

    Returns:
        The groups and the constant (identity) part of the observable.
    """
    groups: list[MeasurementGroup] = []
    constant = 0.0
    n = observable.num_qubits
    for group in observable.group_commuting(qubit_wise=True):
        basis = ["Z"] * n
        coeffs: list[float] = []
        supports: list[tuple[int, ...]] = []
        for pauli, coeff in zip(group.paulis, group.coeffs, strict=True):
            support = tuple(int(q) for q in np.flatnonzero(pauli.x | pauli.z))
            if not support:
                constant += float(np.real(coeff))
                continue
            for q in support:
                basis[q] = "Y" if (pauli.x[q] and pauli.z[q]) else ("X" if pauli.x[q] else "Z")
            coeffs.append(float(np.real(coeff)))
            supports.append(support)
        if supports:
            groups.append(MeasurementGroup(tuple(basis), np.asarray(coeffs), tuple(supports)))
    return groups, constant


@dataclass(frozen=True, slots=True)
class ReadoutCalibration:
    """Per-qubit readout flip probabilities, indexed by logical qubit.

    Attributes:
        e0: ``P(read 1 | prepared 0)``.
        e1: ``P(read 0 | prepared 1)``.
    """

    e0: FloatArray
    e1: FloatArray

    def inverse_matrices(self) -> FloatArray:
        """Per-qubit inverse confusion matrices, shape ``(n, 2, 2)`` (rows: true outcome)."""
        confusion = np.stack(
            [
                np.array([[1.0 - e0, e1], [e0, 1.0 - e1]])
                for e0, e1 in zip(self.e0, self.e1, strict=True)
            ]
        )
        return np.asarray(np.linalg.inv(confusion), dtype=float)

    def weights(self) -> FloatArray:
        """Per-qubit unbiased weights ``v_q[b] = ([1, -1] A_q^{-1})[b]``, shape ``(n, 2)``."""
        det = 1.0 - self.e0 - self.e1
        if np.any(det <= 0.05):
            msg = "readout calibration is too noisy to invert (1 - e0 - e1 <= 0.05)"
            raise ValueError(msg)
        return np.stack([(1.0 - self.e1 + self.e0) / det, -(1.0 + self.e1 - self.e0) / det], axis=1)


def _group_energies(
    bits: BoolArray,
    group: MeasurementGroup,
    weights: FloatArray,
    keep_ones: int | None,
) -> tuple[float, float, int]:
    """Mean energy, its standard error and shots used, from one basis's shots.

    ``bits`` has shape ``(shots, n)``. ``weights[q, b]`` maps bit ``b`` of qubit ``q`` to its
    contribution; the product over a term's support is that term's per-shot value.
    """
    if keep_ones is not None:
        kept = bits[bits.sum(axis=1) == keep_ones]
        if kept.shape[0] == 0:
            log.warning("symmetry filter rejected every shot; using unfiltered shots")
            kept = bits
        bits = kept
    idx = bits.astype(np.intp)
    per_shot = np.zeros(bits.shape[0])
    for coeff, support in zip(group.coeffs, group.supports, strict=True):
        value = np.ones(bits.shape[0])
        for q in support:
            value *= weights[q, idx[:, q]]
        per_shot += coeff * value
    shots = per_shot.size
    sem = float(per_shot.std(ddof=1) / np.sqrt(shots)) if shots > 1 else 0.0
    return float(per_shot.mean()), sem, shots


def _all_bitstrings(n: int) -> BoolArray:
    """``(2**n, n)`` table; row ``i`` holds the bits of ``i`` with bit ``j`` in column ``j``."""
    return ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(bool)


def _apply_inverse(probs: FloatArray, inverses: FloatArray) -> FloatArray:
    """Apply ``inverse_q`` to qubit ``q`` of a distribution indexed by ``sum_j b_j 2**j``."""
    n = inverses.shape[0]
    tensor = probs.reshape([2] * n)  # axis 0 is the most significant bit, qubit n - 1
    for q in range(n):
        axis = n - 1 - q
        tensor = np.moveaxis(np.tensordot(inverses[q], tensor, axes=([1], [axis])), 0, axis)
    return np.asarray(tensor.reshape(-1), dtype=float)


def _sector_energy(
    counts: FloatArray, values: FloatArray, mask: BoolArray, inverses: FloatArray
) -> tuple[float, float]:
    """Correct-then-filter energy of one basis and the kept quasi-probability weight."""
    quasi = _apply_inverse(counts / counts.sum(), inverses)
    kept = float(quasi[mask].sum())
    return float((quasi * values)[mask].sum() / kept), kept


class _SamplerRunner(Protocol):
    def run(self, pubs: list[Any], shots: int) -> tuple[Any, float, str | None]: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, *exc: object) -> None: ...


class _AerSampler:
    """``BackendSamplerV2`` over an ``AerSimulator``, fresh seed per call."""

    def __init__(
        self,
        backend: Any | None,
        method: str,
        seeds: np.random.Generator,
        noise_model: Any | None = None,
    ) -> None:
        from qiskit.primitives import BackendSamplerV2
        from qiskit_aer import AerSimulator

        options: dict[str, Any] = {"method": method}
        if noise_model is not None:
            options["noise_model"] = noise_model
        simulator = (
            AerSimulator(**options)
            if backend is None or isinstance(backend, AerSimulator)
            else AerSimulator.from_backend(backend, **options)
        )
        self._sampler = BackendSamplerV2(backend=simulator)
        self._seeds = seeds

    def run(self, pubs: list[Any], shots: int) -> tuple[Any, float, str | None]:
        self._sampler.options.seed_simulator = int(self._seeds.integers(0, 2**31 - 1))
        start = time.perf_counter()
        result = self._sampler.run(pubs, shots=shots).result()
        return result, time.perf_counter() - start, None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _RuntimeSampler:
    """qiskit-ibm-runtime ``SamplerV2`` through a :class:`RuntimeSession`."""

    def __init__(self, backend: Any, cfg: BackendConfig, seeds: np.random.Generator) -> None:
        self._session = RuntimeSession(backend, cfg, "sampler", seeds)

    def run(self, pubs: list[Any], shots: int) -> tuple[Any, float, str | None]:
        job, result, elapsed = self._session.run(pubs, shots=shots)
        job_id = getattr(job, "job_id", None)
        return (
            result,
            elapsed,
            None if job_id is None else str(job_id() if callable(job_id) else job_id),
        )

    def __enter__(self) -> Self:
        self._session.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self._session.__exit__(None, None, None)


class CountsEnergy:
    """Energy estimator that measures each basis itself and post-processes the shots.

    Args:
        problem: ISA problem; its ``logical_observable`` and ``measurement_qubits`` define what
            is measured and where.
        cfg: Backend configuration; ``kind`` picks Aer or the runtime sampler.
        backend: Target backend object (``None`` for logical problems).
        readout: Calibrate and correct readout errors.
        readout_shots: Shots per calibration circuit.
        symmetry: Keep only shots with exactly ``n/2`` ones in every basis (total-spin singlet
            ansatzes only).
        seed: Seeds the per-call seed stream.
    """

    def __init__(
        self,
        problem: IsaProblem,
        cfg: BackendConfig,
        backend: Any | None,
        *,
        readout: bool = False,
        readout_shots: int = 16384,
        symmetry: bool = False,
        seed: int | None = None,
    ) -> None:
        """Build the measurement circuits; calibration runs in ``__enter__``."""
        if problem.logical_observable is None:
            msg = "CountsEnergy needs a problem built by compile_for_backend (logical observable)"
            raise ValueError(msg)
        if cfg.shots is None:
            msg = "CountsEnergy needs backend.shots"
            raise ValueError(msg)
        self._problem = problem
        self._cfg = cfg
        self._shots = cfg.shots
        self._readout = readout
        self._readout_shots = readout_shots
        self._seeds = np.random.default_rng(seed)
        self._n = problem.logical_observable.num_qubits
        self._keep_ones = self._n // 2 if symmetry else None
        self._groups, self._constant = measurement_plan(problem.logical_observable)
        self._sector_aware = readout and symmetry
        if self._sector_aware:
            if self._n > SECTOR_AWARE_MAX_QUBITS:
                msg = f"readout + symmetry needs <= {SECTOR_AWARE_MAX_QUBITS} qubits, got {self._n}"
                raise ValueError(msg)
            table = _all_bitstrings(self._n)
            self._sector_mask = table.sum(axis=1) == self._n // 2
            signs = 1.0 - 2.0 * table
            self._group_values = [
                np.sum(
                    [
                        c * np.prod(signs[:, list(s)], axis=1)
                        for c, s in zip(g.coeffs, g.supports, strict=True)
                    ],
                    axis=0,
                )
                for g in self._groups
            ]
        self._isa_target = backend if cfg.kind in ("aer_noisy", "fake", "ibm") else None
        self._circuits = [self._measurement_circuit(g.basis) for g in self._groups]
        self._weights = np.tile(_SIGNS, (self._n, 1))
        self.calibration: ReadoutCalibration | None = None
        self._kept_fraction: list[float] = []
        if cfg.kind in ("fake", "ibm"):
            self._runner: _SamplerRunner = _RuntimeSampler(backend, cfg, self._seeds)
        else:
            noisy = cfg.kind == "aer_noisy"
            method: str = cfg.aer_method
            if method == "auto":
                method = (
                    "density_matrix"
                    if noisy and self._n <= DENSITY_MATRIX_MAX_QUBITS
                    else ("statevector" if not noisy else "automatic")
                )
            self._runner = _AerSampler(
                backend if noisy else None,
                method,
                self._seeds,
                aer_noise_model(backend, cfg.noise) if noisy else None,
            )

    # -- circuits -----------------------------------------------------------------------------

    def _suffix(self, basis: Sequence[str] | None, prepare_ones: bool = False) -> QuantumCircuit:
        """Basis rotations (or calibration preparation) and measurement on the physical qubits."""
        width = self._problem.circuit.num_qubits
        suffix = QuantumCircuit(width)
        physical = self._problem.measurement_qubits
        for logical, phys in enumerate(physical):
            if prepare_ones:
                suffix.x(phys)
            elif basis is not None and basis[logical] == "X":
                suffix.h(phys)
            elif basis is not None and basis[logical] == "Y":
                suffix.sdg(phys)
                suffix.h(phys)
        if self._isa_target is not None and suffix.size() > 0:
            # Translate only the single-qubit suffix: re-optimising the body would undo folding.
            pm = generate_preset_pass_manager(
                optimization_level=1,
                backend=self._isa_target,
                initial_layout=list(range(width)),
            )
            suffix = pm.run(suffix)
        creg = ClassicalRegister(self._n, "meas")
        suffix.add_register(creg)
        for logical, phys in enumerate(physical):
            suffix.measure(phys, creg[logical])
        return suffix

    def _measurement_circuit(self, basis: Sequence[str]) -> QuantumCircuit:
        body = self._problem.circuit.copy()
        suffix = self._suffix(basis)
        body.add_register(suffix.cregs[0])
        return body.compose(suffix, qubits=range(body.num_qubits), clbits=suffix.clbits)

    # -- lifecycle ----------------------------------------------------------------------------

    @property
    def description(self) -> dict[str, Any]:
        """Backend, mitigation settings and calibration."""
        return {
            "kind": f"counts:{self._cfg.kind}",
            "backend": self._cfg.label(),
            "shots": self._shots,
            "readout_mitigation": self._readout,
            "readout_shots": self._readout_shots if self._readout else 0,
            "symmetry_filter": self._keep_ones is not None,
            "readout_symmetry_order": "correct_then_filter" if self._sector_aware else None,
            "mean_kept_fraction": float(np.mean(self._kept_fraction))
            if self._kept_fraction
            else None,
            "readout_e0": None if self.calibration is None else self.calibration.e0.tolist(),
            "readout_e1": None if self.calibration is None else self.calibration.e1.tolist(),
            "std": "exact standard error of the per-shot estimator",
        }

    def __enter__(self) -> Self:
        """Start the sampler and, if requested, calibrate readout."""
        self._runner.__enter__()
        if self._readout:
            self.calibration = self._calibrate()
            self._weights = self.calibration.weights()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop the sampler."""
        self._runner.__exit__(exc_type, exc, tb)

    def _calibrate(self) -> ReadoutCalibration:
        width = self._problem.circuit.num_qubits
        zeros = QuantumCircuit(width, self._n)
        zeros.compose(self._suffix(None), inplace=True)
        ones = QuantumCircuit(width, self._n)
        ones.compose(self._suffix(None, prepare_ones=True), inplace=True)
        result, _, _ = self._runner.run([(zeros,), (ones,)], self._readout_shots)
        b0 = next(iter(result[0].data.values())).to_bool_array(order="little")
        b1 = next(iter(result[1].data.values())).to_bool_array(order="little")
        cal = ReadoutCalibration(e0=b0.mean(axis=0), e1=1.0 - b1.mean(axis=0))
        log.info(
            "readout calibration: mean P(1|0)=%.4f, mean P(0|1)=%.4f",
            float(cal.e0.mean()),
            float(cal.e1.mean()),
        )
        return cal

    # -- evaluation ---------------------------------------------------------------------------

    def _sector_aware_group(self, bits: BoolArray, g: int) -> tuple[float, float, float]:
        """Correct-then-filter energy, bootstrap standard error and kept quasi weight."""
        assert self.calibration is not None
        index = (bits.astype(np.int64) << np.arange(self._n)).sum(axis=1)
        counts = np.bincount(index, minlength=2**self._n).astype(float)
        inverses = self.calibration.inverse_matrices()
        values, mask = self._group_values[g], self._sector_mask
        energy, kept = _sector_energy(counts, values, mask, inverses)
        shots = int(counts.sum())
        resampled = [
            _sector_energy(
                self._seeds.multinomial(shots, counts / shots).astype(float), values, mask, inverses
            )[0]
            for _ in range(_BOOTSTRAP)
        ]
        return energy, float(np.std(resampled, ddof=1)), kept

    def evaluate(self, params: FloatArray, *, shots: int | None = None) -> EnergyBatch:
        """Measure every basis for each parameter row and post-process the shots."""
        params = np.atleast_2d(np.asarray(params, dtype=float))
        n_shots = shots or self._shots
        pubs: list[Any] = [
            (circuit, params) if circuit.num_parameters else (circuit,)
            for circuit in self._circuits
        ]
        result, elapsed, job_id = self._runner.run(pubs, n_shots)
        k = params.shape[0]
        energies = np.full(k, self._constant)
        variances = np.zeros(k)
        for g, group in enumerate(self._groups):
            data = next(iter(result[g].data.values()))
            for row in range(k):
                row_bits = data[row] if data.ndim else data
                bits = row_bits.to_bool_array(order="little")
                if self._sector_aware:
                    mean, sem, kept = self._sector_aware_group(bits, g)
                    self._kept_fraction.append(kept)
                else:
                    mean, sem, used = _group_energies(bits, group, self._weights, self._keep_ones)
                    if self._keep_ones is not None:
                        self._kept_fraction.append(used / bits.shape[0])
                energies[row] += mean
                variances[row] += sem**2
        return EnergyBatch(
            energies=energies,
            stds=np.sqrt(variances),
            shots=n_shots,
            job_id=job_id,
            wall_time_s=elapsed,
        )
