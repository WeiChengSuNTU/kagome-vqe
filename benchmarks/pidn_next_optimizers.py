"""Test C: optimisers at equal shots under compiled hardware noise (FakeKingston).

Usage::

    uv run python benchmarks/pidn_next_optimizers.py plan
    uv run python benchmarks/pidn_next_optimizers.py run METHOD SEED [--threads 2]
    uv run python benchmarks/pidn_next_optimizers.py direction POINT [--fold 1|3|5]
    uv run python benchmarks/pidn_next_optimizers.py report [--out DIR]

Test A of ``docs/experiments/graph_hva_pidn_next.md`` found that compiled noise on FakeKingston
turns the energy gradient (cosine with the noise-free gradient down to ~0.77), unlike the PRB 106
layer model; that is the noise model further PIDN work should use. Simulating it exactly costs
~10-25 s per energy for the 12-qubit 2x6 problem (a 60-iteration gradient descent: ~1 day), so
this comparison uses the **2x5 rectangle** (10 qubits, ``p = 2``, 28 per-gate parameters, PRB 105
schedule), routed on a 10-qubit path of FakeKingston's best-calibrated hexagon.

**Estimator.** Per energy: the exact density matrix of the compiled circuit under the device
noise model (Aer, gates and relaxation), its outcome distribution in each of the three uniform
bases, the device's per-qubit readout flips, 1024 shots per basis sampled from that distribution,
then readout correction (exact per-qubit inverse) and symmetry filtering (``n/2`` ones), the
correct-then-filter estimator of ``LogicalNoiseEnergy`` ("readout + symmetry", the M2 choice).
Statistically identical to sampling the circuit with Aer, at one 10-qubit density matrix per
energy. Not modelled: the finite readout calibration.

**Methods**, same seeds and initial points, the same budget of 3,421 energies (= 60 iterations
of parameter-shift gradient descent; x 3 bases x 1024 shots):

``gd``     parameter-shift gradient descent (exchange shift rule), ``lr = 0.06``
           (``lr * lambda_max(x0) = 1.5``, as for the 2x6 runs).
``gdm``    the same with heavy-ball momentum ``beta = 0.9``, ``lr = 0.006`` (same terminal step).
``mgd``    model gradient descent (Sung et al. 2020), 57 new energies per iteration (as one
           gradient), radius and rate from the seed-101 pilot.
``spsa``   calibrated SPSA, first step and step cap 0.2, ``c = 0.1``.
``pidn``   noise-aware PIDN (chi-square physics loss stopped at the noise floor), Stage I = the
           first 10 iterations of ``gd``, a measured gradient every 5 steps, continued until the
           budget is spent.

Runs are resumable: every objective call is cached under ``results/pidn_next_cache/`` and its
sampling seed is derived from (method, seed, call index), so a restarted process replays the
cached calls and continues identically.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import noise_direction as nd
from m2_exact_limits import basis_probabilities, basis_statistic

from kagome_vqe.mitigation.counts import _apply_inverse

log = logging.getLogger("pidn_next")

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "pidn_next_cache"
ROWS, COLS, REPS = 2, 5, 2
SHOTS, FINAL_SHOTS = 1024, 16384
GD_ITERATIONS = 60
LR = 0.06
X0_SCALE = 0.1
BASES = "XYZ"
METHODS = ("gd", "gdm", "mgd", "spsa", "pidn", "gd_exact")
MGD = {"sample_radius": 0.5, "lr": LR}  # seed-101 pilot: 0.1 -> 3.5 %, 0.3 -> 1.1 %, 0.5 -> 0.9 %
PIDN_T_INIT, PIDN_ANCHOR_EVERY = 10, 5


# -- the problem -----------------------------------------------------------------------------------


class Problem:
    """2x5, p = 2 on a FakeKingston hexagon path: compiled circuit, noise, readout, references."""

    def __init__(self, threads: int = 2, fold: int = 1) -> None:  # noqa: PLR0915 (linear set-up)
        """Compile (deterministic) and set up the exact references."""
        from qiskit import transpile
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.transpiler import CouplingMap
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel
        from scipy.sparse.linalg import eigsh

        from kagome_vqe.ansatz.unshare import unshare_parameters
        from kagome_vqe.backends import resolve_fake_backend
        from kagome_vqe.config import AnsatzConfig, LatticeConfig
        from kagome_vqe.hamiltonian import heisenberg_hamiltonian
        from kagome_vqe.mapping import map_circuit

        lattice = LatticeConfig.model_validate(
            {"kind": "kagome_bm_rectangle", "rows": ROWS, "cols": COLS}
        ).build()
        ansatz = AnsatzConfig.model_validate(
            {"type": "graph_hva", "reps": REPS, "schedule": "prb105"}
        ).build(lattice)
        self.hamiltonian: SparsePauliOp = heisenberg_hamiltonian(lattice)
        self.logical = ansatz.build()
        self.n = ansatz.num_parameters
        _, self.parameter_map = unshare_parameters(self.logical, exchange_gates=True)
        self._h = self.hamiltonian.to_matrix(sparse=True)
        self.e0 = float(eigsh(self._h, k=1, which="SA")[0][0])
        backend = resolve_fake_backend("FakeKingston")
        ring = nd._ring_kingston(backend)
        routed = map_circuit("generic", lattice, self.logical, self.hamiltonian).circuit
        best: tuple[int, Any, list[int]] | None = None
        for start in range(len(ring)):  # the 10-qubit path of the ring with the fewest CZ
            path = [ring[(start + i) % len(ring)] for i in range(lattice.num_sites)]
            edges = list(itertools.pairwise(path))
            isa = transpile(
                routed,
                backend=backend,
                coupling_map=CouplingMap(edges + [(b, a) for a, b in edges]),
                initial_layout=path,
                optimization_level=3,
                seed_transpiler=nd.TRANSPILE_SEED,
            )
            count = int(isa.count_ops().get("cz", 0))
            if best is None or count < best[0]:
                best = (count, isa, path)
        assert best is not None
        cz, isa, path = best
        active = nd._active(isa)
        local = {q: i for i, q in enumerate(active)}
        noise_model = NoiseModel.from_backend(backend)
        readout = nd._readout(noise_model, active)
        circuit = nd._compact_circuit(isa, local)
        if fold > 1:
            circuit = nd.fold_two_qubit_gates(circuit, fold)
        self.circuit = circuit
        self.fingerprint = nd.circuit_fingerprint(circuit)
        self.circuit.save_density_matrix()
        self.terms = [
            (c, letter, local[a], local[b])
            for c, letter, a, b in nd._terms(self.hamiltonian.apply_layout(isa.layout))
        ]
        self.width = len(active)
        q = np.array([readout[p] for p in active])  # (p(1|0), p(0|1)) per local qubit
        self.confusion = np.array([[[1 - a, b], [a, 1 - b]] for a, b in q])
        self.inverse = np.linalg.inv(self.confusion)
        bits = ((np.arange(2**self.width)[:, None] >> np.arange(self.width)) & 1).astype(float)
        z = 1 - 2 * bits
        self.diagonals = {
            b: sum(
                (c * z[:, i] * z[:, j] for c, letter, i, j in self.terms if letter == b),
                start=np.zeros(2**self.width),
            )
            for b in BASES
        }
        self._sim = AerSimulator(
            method="density_matrix",
            noise_model=nd._compact_noise(noise_model, local),
            max_parallel_threads=threads,
        )
        self.info = {
            "device": str(backend.name),
            "path": path,
            "cz": cz,
            "fold": fold,
            "qubits": self.width,
            "parameters": self.n,
            "readout_error_mean": float(q.mean()),
            "fingerprint": self.fingerprint,
            "e0": self.e0,
        }

    def distributions(self, theta: np.ndarray) -> dict[str, np.ndarray]:
        """Pre-readout outcome distribution per basis (exact)."""
        bound = self.circuit.assign_parameters(np.asarray(theta, dtype=float))
        result = self._sim.run(bound).result()
        nd.check_success(result)
        rho = np.asarray(result.data()["density_matrix"], dtype=complex)
        return {b: np.clip(basis_probabilities(rho, b * self.width), 0, None) for b in BASES}

    def exact_energies(self, theta: np.ndarray) -> tuple[float, float, float]:
        """Infinite shots: ``(readout-corrected, unmitigated, readout + symmetry)`` energies."""
        probs = self.distributions(theta)
        gate = sum(float(probs[b] @ self.diagonals[b]) for b in BASES)
        read = {b: _apply_inverse(probs[b], self.confusion) for b in BASES}
        raw = sum(float(read[b] @ self.diagonals[b]) for b in BASES)
        rs = sum(
            basis_statistic(
                read[b], self.diagonals[b], symmetry=True, inverses=self.inverse, n=self.width
            )[0]
            for b in BASES
        )
        return gate, raw, rs

    def ideal_energy(self, theta: np.ndarray) -> float:
        """Noise-free energy of the logical circuit."""
        from qiskit.quantum_info import Statevector

        psi = Statevector(self.logical.assign_parameters(np.asarray(theta, dtype=float))).data
        return float(np.real(np.vdot(psi, self._h @ psi)))

    def ideal_gradient(self, theta: np.ndarray) -> np.ndarray:
        """Exchange shift rule on the logical circuit (exact)."""
        out = np.empty(self.n)
        for k in range(self.n):
            e = np.zeros(self.n)
            e[k] = np.pi / 4
            out[k] = self.ideal_energy(theta + e) - self.ideal_energy(theta - e)
        return out


class SampledEnergy:
    """Readout + symmetry energies from sampled shots; every call cached (resumable)."""

    def __init__(self, problem: Problem, tag: str, seed: int, shots: int = SHOTS) -> None:
        """``tag`` names the cache directory; ``seed`` and the call index seed the sampling."""
        self.problem, self.shots, self.seed = problem, shots, seed
        self.dir = CACHE / tag
        self.dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.energies = 0
        self.trace: list[dict[str, Any]] = []

    def _one(self, probs: dict[str, np.ndarray], rng: np.random.Generator) -> tuple[float, float]:
        p = self.problem
        energy, variance = 0.0, 0.0
        for b in BASES:
            read = np.clip(_apply_inverse(probs[b], p.confusion), 0, None)
            counts = rng.multinomial(self.shots, read / read.sum())
            value, var, _ = basis_statistic(
                counts / self.shots, p.diagonals[b], symmetry=True, inverses=p.inverse, n=p.width
            )
            energy += value
            variance += var
        return energy, math.sqrt(max(variance, 0.0) / self.shots)

    def batch(self, xs: np.ndarray) -> np.ndarray:
        """Energies of the rows of ``xs`` (cached per call)."""
        xs = np.atleast_2d(np.asarray(xs, dtype=float))
        path = self.dir / f"call{self.calls:05d}.npz"
        if path.exists():
            z = np.load(path)
            if z["x"].shape != xs.shape or not np.allclose(z["x"], xs):
                msg = f"{path}: cached call does not match the replay"
                raise RuntimeError(msg)
            if str(z["fp"]) != self.problem.fingerprint:
                msg = f"{path}: cached with another compilation ({z['fp']})"
                raise RuntimeError(msg)
            values, stds = z["e"], z["s"]
        else:
            if nd.out_of_time():
                log.info("%s: deadline, stopping before call %d", self.dir.name, self.calls)
                sys.exit(0)
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.calls]))
            pairs = [self._one(self.problem.distributions(x), rng) for x in xs]
            values = np.array([v for v, _ in pairs])
            stds = np.array([s for _, s in pairs])
            np.savez(
                path.with_suffix(".tmp.npz"), x=xs, e=values, s=stds, fp=self.problem.fingerprint
            )
            path.with_suffix(".tmp.npz").rename(path)
        self.calls += 1
        self.energies += len(xs)
        self.trace.append({"call": self.calls - 1, "energies": self.energies, "std": stds.mean()})
        return np.asarray(values, dtype=float)

    def __call__(self, x: np.ndarray) -> float:
        """One energy."""
        return float(self.batch(np.asarray(x)[None, :])[0])

    @property
    def last_stds(self) -> float:
        """Mean standard error of the last call."""
        return float(self.trace[-1]["std"]) if self.trace else math.nan


class ExactEnergy:
    """Noise-free energies (statevector), counted like :class:`SampledEnergy`.

    The reference: gradient descent with the same steps and neither noise nor shots.
    """

    def __init__(self, problem: Problem) -> None:
        """Wrap ``problem.ideal_energy``."""
        self.problem = problem
        self.energies = 0
        self.trace: list[dict[str, Any]] = []

    def batch(self, xs: np.ndarray) -> np.ndarray:
        """Exact energies of the rows of ``xs``."""
        xs = np.atleast_2d(np.asarray(xs, dtype=float))
        self.energies += len(xs)
        self.trace.append({"call": len(self.trace), "energies": self.energies, "std": 0.0})
        return np.array([self.problem.ideal_energy(x) for x in xs])

    def __call__(self, x: np.ndarray) -> float:
        """One energy."""
        return float(self.batch(np.asarray(x)[None, :])[0])


# -- methods ---------------------------------------------------------------------------------------


def budget(problem: Problem) -> int:
    """Energies of 60 parameter-shift iterations (with the current point) plus the final one."""
    return GD_ITERATIONS * (2 * problem.parameter_map.num_occurrences + 1) + 1


@dataclass(slots=True)
class Visited:
    """Iterates and the energies spent when each was reached."""

    energies: list[int]
    thetas: list[np.ndarray]


def _x0(problem: Problem, seed: int) -> np.ndarray:
    return X0_SCALE * np.random.default_rng(seed).uniform(0, 1, problem.n)


def _run_optimizer(opt: Any, objective: SampledEnergy, x0: np.ndarray) -> tuple[Visited, Any]:
    from kagome_vqe.optimizers import IterationInfo

    path = Visited([0], [x0.copy()])

    def record(info: IterationInfo) -> None:
        path.energies.append(objective.energies)
        path.thetas.append(np.array(info.x, dtype=float))

    result = opt.minimize(objective, x0, callback=record)
    return path, result


def _gd(problem: Problem, objective: SampledEnergy, seed: int, momentum: float = 0.0) -> Any:
    from kagome_vqe.optimizers import GradientDescent, ParameterShift

    lr = LR * (1 - momentum)
    opt = GradientDescent(
        maxiter=GD_ITERATIONS,
        lr=lr,
        gradient=ParameterShift(problem.parameter_map, include_value=True),
        momentum=momentum,
    )
    return _run_optimizer(opt, objective, _x0(problem, seed)), opt


def _mgd(problem: Problem, objective: SampledEnergy, seed: int, radius: float, lr: float) -> Any:
    from kagome_vqe.optimizers import ModelGradientDescent

    k = 2 * problem.parameter_map.num_occurrences + 1
    opt = ModelGradientDescent(
        maxiter=(budget(problem) - 1) // k,
        lr=lr,
        sample_radius=radius,
        sample_points=k,
        seed=np.random.SeedSequence([seed, 77]).generate_state(1)[0],
    )
    return _run_optimizer(opt, objective, _x0(problem, seed)), opt


def _spsa(problem: Problem, objective: SampledEnergy, seed: int) -> Any:
    from kagome_vqe.optimizers import Spsa

    samples = 25
    opt = Spsa(
        maxiter=(budget(problem) - 2 * samples - 1) // 2,
        perturbation=0.1,
        calibrate=True,
        calibration_samples=samples,
        target_magnitude=0.2,
        max_step_norm=0.2,
        seed=np.random.SeedSequence([seed, 78]).generate_state(1)[0],
    )
    return _run_optimizer(opt, objective, _x0(problem, seed)), opt


def _pidn(problem: Problem, seed: int) -> tuple[Visited, dict[str, Any]]:  # noqa: PLR0915 (one loop)
    """Noise-aware PIDN: Stage I from the cached ``gd`` run, then surrogate steps + anchors."""
    from kagome_vqe.optimizers import ParameterShift
    from kagome_vqe.pidn.model import PidnModelConfig, build_batch
    from kagome_vqe.pidn.training import PidnEnsemble, PidnTrainingConfig

    stage1 = SampledEnergy(problem, f"gd_s{seed}", seed)
    (_, _), gd = _gd_replay(problem, stage1, seed)
    records = gd.records[:PIDN_T_INIT]
    thetas = [r["theta"] for r in records]
    grads = [np.asarray(r["grad"], dtype=float) for r in records]
    values = [float(r["value"]) for r in records]
    noisy = list(values)  # readout + symmetry energy at theta_t (same estimator as the targets)
    stds = [float(t["std"]) for t in stage1.trace[:PIDN_T_INIT]]
    c2 = np.bincount(
        problem.parameter_map.occurrence_to_param,
        weights=problem.parameter_map.occurrence_coefficients**2,
        minlength=problem.n,
    )
    variances = [2 * s**2 * c2 for s in stds]
    lr = LR
    theta = thetas[-1] - lr * grads[-1]
    used = records[-1]["nfev"]
    path = Visited([0, *[r["nfev"] for r in records]], [thetas[0], *thetas[1:], theta])
    stage3 = SampledEnergy(problem, f"pidn_s{seed}", seed + 1000)
    model = PidnModelConfig()
    training = PidnTrainingConfig(noise_aware=True)
    surrogate = PidnEnsemble(model, training)
    window = model.history

    fit = surrogate.fit(_anchor_batch(thetas, theta, noisy, values, grads, variances, window))
    log.info("pidn s%d stage II: %s", seed, {k: fit[k] for k in ("epochs_selected",)})
    total = budget(problem)
    estimator = ParameterShift(problem.parameter_map, include_value=True)
    anchors, step, cosines = 0, 0, []
    k = 2 * problem.parameter_map.num_occurrences + 1
    while True:
        step += 1
        is_anchor = step % PIDN_ANCHOR_EVERY == 0
        cost = k if is_anchor else 1
        if used + stage3.energies + cost + 1 > total:
            break
        if is_anchor:
            est = estimator.estimate(stage3, theta)
            g, value = np.asarray(est.grad, dtype=float), float(est.value)
            std = stage3.last_stds
        else:
            value = stage3(theta)
        query = build_batch(
            np.vstack([*thetas, theta]),
            np.asarray([*noisy, value]),
            window=window,
            rows=[len(thetas)],
        )
        _, predicted, _ = surrogate.predict(query)
        s_grad = np.asarray(predicted[0], dtype=float)
        if is_anchor:
            cosines.append(float(s_grad @ g / (np.linalg.norm(s_grad) * np.linalg.norm(g))))
            anchors += 1
        else:
            g = s_grad
        thetas.append(theta.copy())
        noisy.append(value)
        new = theta - lr * g
        if is_anchor:
            values.append(value)
            grads.append(g)
            variances.append(2 * std**2 * c2)
        theta = new
        path.energies.append(used + stage3.energies)
        path.thetas.append(theta.copy())
        if is_anchor:
            surrogate.refresh(_anchor_batch(thetas, theta, noisy, values, grads, variances, window))
        if not np.all(np.isfinite(theta)):
            break
    return path, {"anchors": anchors, "anchor_cosine_mean": float(np.mean(cosines or [np.nan]))}


def _anchor_batch(
    thetas: list[np.ndarray],
    theta: np.ndarray,
    noisy: list[float],
    values: list[float],
    grads: list[np.ndarray],
    variances: list[np.ndarray],
    window: int,
) -> Any:
    """Training rows: Stage I and every anchor (rows whose targets were measured)."""
    from kagome_vqe.pidn.model import build_batch

    measured = [i for i in range(len(thetas)) if i < PIDN_T_INIT] + [
        i for i in range(PIDN_T_INIT, len(thetas)) if (i - PIDN_T_INIT + 1) % PIDN_ANCHOR_EVERY == 0
    ]
    length = len(thetas)
    full_v = np.full(length, np.nan)
    full_g = np.full((length, thetas[0].size), np.nan)
    full_var = np.full((length, thetas[0].size), np.nan)
    full_v[measured] = values
    full_g[measured] = np.vstack(grads)
    full_var[measured] = np.vstack(variances)
    steps = np.diff(np.vstack([*thetas, theta]), axis=0)
    return build_batch(
        np.vstack(thetas),
        np.asarray(noisy),
        window=window,
        values=full_v,
        gradients=full_g,
        steps=steps,
        rows=measured,
        gradient_variances=full_var,
    )


def _gd_replay(problem: Problem, objective: SampledEnergy, seed: int) -> Any:
    return _gd(problem, objective, seed)


def run(method: str, seed: int, threads: int, tag: str | None = None, **extra: Any) -> Path:
    """One (method, seed); writes ``results/pidn_next_cache/<tag>/result.npz``."""
    tag = tag or f"{method}_s{seed}"
    out = CACHE / tag / "result.npz"
    if out.exists():
        log.info("%s: done", tag)
        return out
    tick = time.perf_counter()
    problem = Problem(threads)
    info: dict[str, Any] = {}
    if method == "pidn":
        path, info = _pidn(problem, seed)
        energies_used = path.energies[-1]
    else:
        objective = SampledEnergy(problem, tag, seed)
        if method == "gd":
            (path, _), _ = _gd(problem, objective, seed)
        elif method == "gdm":
            (path, _), _ = _gd(problem, objective, seed, momentum=0.9)
        elif method == "mgd":
            radius = float(extra.get("radius", MGD["sample_radius"]))
            lr = float(extra.get("lr", MGD["lr"]))
            (path, _), _ = _mgd(problem, objective, seed, radius, lr)
            info = {"radius": radius, "lr": lr}
        elif method == "spsa":
            (path, _), _ = _spsa(problem, objective, seed)
        elif method == "gd_exact":
            objective = ExactEnergy(problem)  # type: ignore[assignment]
            (path, _), _ = _gd(problem, objective, seed)  # type: ignore[arg-type]
        else:
            msg = f"unknown method {method}"
            raise ValueError(msg)
        energies_used = objective.energies
    final_theta = path.thetas[-1]
    final = SampledEnergy(problem, f"{tag}_final", seed + 5000, shots=FINAL_SHOTS)
    final_value = final(final_theta)
    ideal = np.array([problem.ideal_energy(t) for t in path.thetas])
    np.savez(
        out,
        energies=np.asarray(path.energies),
        thetas=np.vstack(path.thetas),
        ideal=ideal,
        e0=problem.e0,
        final_measured=final_value,
        final_std=final.last_stds,
        energies_used=energies_used,
        info=json.dumps({"method": method, "seed": seed, **problem.info, **info}),
    )
    log.info(
        "%s: final noise-free rel. err %.4f, measured %.4f, %d energies, %.0f s",
        tag,
        abs(ideal[-1] - problem.e0) / abs(problem.e0),
        final_value,
        energies_used,
        time.perf_counter() - tick,
    )
    return out


# -- the direction test on this instance ------------------------------------------------------


def direction(point: str, fold: int, threads: int) -> Path:
    """Exact gradients at an iterate of ``gd_s1`` (``it<k>``): FD, the shift rule, readout."""
    out = CACHE / "direction" / f"{point}_fold{fold}.npz"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    it = int(point.removeprefix("it"))
    res = np.load(CACHE / "gd_s1" / "result.npz")
    theta = res["thetas"][it]
    problem = Problem(threads, fold=fold)
    h = nd.H_FD
    rows = [theta]
    for k in range(problem.n):
        for shift in (h, -h, np.pi / 4, -np.pi / 4):
            t = theta.copy()
            t[k] += shift
            rows.append(t)
    values = np.array([problem.exact_energies(t) for t in rows])
    np.savez(
        out,
        theta=theta,
        values=values,
        fp=problem.fingerprint,
        g_ideal=problem.ideal_gradient(theta),
        e_ideal=problem.ideal_energy(theta),
    )
    return out


def _direction_rows() -> list[dict[str, Any]]:
    rows = []
    folds: dict[str, dict[int, Any]] = {}
    for path in sorted((CACHE / "direction").glob("*.npz")):
        point, fold = path.stem.rsplit("_fold", 1)
        z = np.load(path)
        v = z["values"]  # (1 + 4n, 3): readout-corrected, unmitigated, readout + symmetry
        n = z["g_ideal"].size
        quad = v[1:].reshape(n, 4, 3)
        fd = (quad[:, 0] - quad[:, 1]) / (2 * nd.H_FD)
        ps = quad[:, 2] - quad[:, 3]
        gi = z["g_ideal"]
        folds.setdefault(point, {})[int(fold)] = (v[0, 0], fd[:, 0], fd[:, 2])
        rows.append(
            {
                "point": point,
                "fold": int(fold),
                "e_ideal": float(z["e_ideal"]),
                "bias": float(v[0, 0] - z["e_ideal"]),
                "cos_fd": nd._cos(fd[:, 0], gi),
                "cos_fd_readout": nd._cos(fd[:, 1], gi),
                "cos_fd_rs": nd._cos(fd[:, 2], gi),
                "bias_rs": float(v[0, 2] - z["e_ideal"]),
                "cos_shift_rule": nd._cos(ps[:, 0], gi),
                "cos_shift_rule_rs": nd._cos(ps[:, 2], gi),
                "shift_rule_vs_fd": nd._cos(ps[:, 0], fd[:, 0]),
                "g_ideal_norm": float(np.linalg.norm(gi)),
                "g_noisy_norm": float(np.linalg.norm(fd[:, 0])),
            }
        )
    for point, by_fold in folds.items():
        if all(f in by_fold for f in nd.FOLDS):
            gi = np.load(CACHE / "direction" / f"{point}_fold1.npz")["g_ideal"]
            for name, degree in (("zne_linear", 1), ("zne_quadratic", 2)):
                w = nd.richardson_weights(nd.FOLDS, degree)
                g = w @ np.vstack([by_fold[f][1] for f in nd.FOLDS])
                g_rs = w @ np.vstack([by_fold[f][2] for f in nd.FOLDS])
                rows.append(
                    {
                        "point": point,
                        "fold": name,
                        "cos_fd": nd._cos(g, gi),
                        "cos_fd_rs": nd._cos(g_rs, gi),
                    }
                )
    return rows


# -- report ----------------------------------------------------------------------------------------

_LABELS = {
    "gd": "gradient descent",
    "gdm": "GD + momentum 0.9",
    "mgd": "model gradient descent",
    "spsa": "SPSA, cap 0.2",
    "pidn": "noise-aware PIDN, anchor every 5",
    "gd_exact": "GD without noise or shots (reference)",
}


def _plot(curves: pd.DataFrame, path: Path) -> None:
    """Noise-free relative error against circuit executions, one panel, one line per seed."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = dict(zip(METHODS, plt.rcParams["axes.prop_cycle"].by_key()["color"], strict=False))
    for method in METHODS:
        sub = curves[curves["method"] == method]
        for i, (_, run) in enumerate(sub.groupby("seed")):
            ax.plot(
                run["executions"],
                run["rel_err"],
                color=colors[method],
                alpha=0.3 if method == "spsa" else 0.85,
                lw=0.8 if method == "spsa" else 1.4,
                zorder=1 if method == "spsa" else 2,
                label=_LABELS[method] if i == 0 else None,
            )
    ax.set_yscale("log")
    ax.set_xlabel("circuit executions (3 bases per energy, 1024 shots each)")
    ax.set_ylabel("noise-free relative error of the parameters")
    ax.set_title("2x5, p = 2, FakeKingston compiled noise, readout + symmetry; seeds 1-3")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(out: Path) -> None:
    """Tables of every finished run and the direction test."""
    out.mkdir(parents=True, exist_ok=True)
    runs, curves = [], []
    for path in sorted(CACHE.glob("*/result.npz")):
        z = np.load(path)
        info = json.loads(str(z["info"]))
        e0 = float(z["e0"])
        rel = np.abs(z["ideal"] - e0) / abs(e0)
        runs.append(
            {
                "run": path.parent.name,
                "method": info["method"],
                "seed": info["seed"],
                "final_rel_err": float(rel[-1]),
                "best_rel_err": float(rel.min()),
                "final_measured": float(z["final_measured"]),
                "final_measured_rel_err": float(abs(z["final_measured"] - e0) / abs(e0)),
                "final_std": float(z["final_std"]),
                "energies": int(z["energies_used"]),
                "circuit_executions": 3 * int(z["energies_used"]),
                **{
                    k: info[k]
                    for k in ("radius", "lr", "anchors", "anchor_cosine_mean")
                    if k in info
                },
            }
        )
        curves += [
            {
                "run": path.parent.name,
                "method": info["method"],
                "seed": info["seed"],
                "energies": int(e),
                "executions": 3 * int(e),
                "rel_err": float(r),
            }
            for e, r in zip(z["energies"], rel, strict=True)
        ]
    frame = pd.DataFrame(runs)
    frame.to_csv(out / "runs.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "curves.csv", index=False)
    if curves:
        main = pd.DataFrame(curves)
        _plot(main[main["seed"] != 101], out / "pidn_next_optimizers.png")  # no pilots
    direction_rows = _direction_rows()
    pd.DataFrame(direction_rows).to_csv(out / "direction.csv", index=False)
    lines = ["# Test C (2x5, FakeKingston, readout + symmetry)", "", nd._md(frame)]
    if direction_rows:
        lines += [
            "",
            "## Exact gradient direction on this instance",
            "",
            nd._md(pd.DataFrame(direction_rows)),
        ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    sys.stdout.write("\n".join(lines) + "\n")


def main() -> None:
    """CLI."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    r = sub.add_parser("run")
    r.add_argument("method", choices=METHODS)
    r.add_argument("seed", type=int)
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--radius", type=float, default=None)
    r.add_argument("--lr", type=float, default=None)
    r.add_argument("--tag", default=None)
    d = sub.add_parser("direction")
    d.add_argument("point")
    d.add_argument("--fold", type=int, default=1)
    d.add_argument("--threads", type=int, default=2)
    rep = sub.add_parser("report")
    rep.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.cmd == "plan":
        problem = Problem(1)
        tick = time.perf_counter()
        problem.distributions(_x0(problem, 1))
        seconds = time.perf_counter() - tick
        b = budget(problem)
        sys.stdout.write(
            f"{json.dumps(problem.info)}\nbudget {b} energies = {3 * b} circuits x {SHOTS} shots;"
            f" {seconds:.2f} s per energy (1 thread) -> {b * seconds / 60:.0f} min per run\n"
        )
    elif args.cmd == "run":
        extra = {k: v for k, v in (("radius", args.radius), ("lr", args.lr)) if v is not None}
        run(args.method, args.seed, args.threads, args.tag, **extra)
    elif args.cmd == "direction":
        direction(args.point, args.fold, args.threads)
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        report(args.out or ROOT / "results" / f"{stamp}-graph_hva_pidn_next_c")


if __name__ == "__main__":
    main()
