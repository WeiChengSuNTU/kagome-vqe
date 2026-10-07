# Roadmap: closing the noisy-energy gap

Status: proposal, October 2026. Owner: the kagome-vqe group.

This document records what limits the precision of noisy VQE runs today, the improvement
directions discussed in the lab, advice on each, and an ordered plan with a benchmark check
for every step. It is meant to be updated as milestones land: tick the exit criteria, link the
results directories, and move findings into `docs/physics.md` once they are settled.

## 1. Where we are

### 1.1 Measured baseline

| setting | best energy | relative error | source |
|---|---|---|---|
| exact statevector, HVA + ADAM | -18.0000 | 0.000 % | `configs/benchmark.toml` |
| exact statevector, RVB + SPSA | -17.8386 | 0.897 % | `configs/benchmark.toml` |
| noisy FakeGuadalupeV2, all four combos | -14.01 to -14.10 | 21.7 to 22.2 % | `configs/benchmark_fake_guadalupe.toml` |

Full tables are in the README.

### 1.2 Diagnosis: the noisy runs converged to a biased floor

The optimiser is not the bottleneck. Evaluating the circuits at the *exact* ground-state
parameters on the same noise model gives the same plateau:

| ansatz | CX gates | noise-free | full noise | readout error only | gate + T1/T2 only |
|---|---|---|---|---|---|
| HVA | 66 | -18.000 | -13.973 | -18.000 | -13.973 |
| RVB | 66 | -18.000 | -13.900 | -18.000 | -13.900 |

Noise turns the prepared state into a mixture. Under a depolarising model,
`E_measured ~= (1 - p) * E_ideal`, because every Pauli term averages to zero on the maximally
mixed state. A reading of -14 corresponds to roughly 78 % of the state surviving 66 CX gates.
More iterations cannot remove this: it is a bias, not a convergence failure. The variational
principle also holds for mixed states, so a noisy reading below -18 signals an artefact.

### 1.3 Known defects in the noisy simulation

These were found during the diagnosis and must be fixed before any benchmark below is trusted.

1. **Shots and readout error are ignored by the `aer_ideal` and `aer_noisy` kinds.**
   `qiskit_aer.primitives.EstimatorV2` computes expectation values with
   `save_expectation_value` instead of sampling, so `backend.shots` has no effect, reported
   `stds` are zero, and readout error never applies. The noisy benchmark is therefore
   optimistic, and the unit test `test_aer_ideal_is_shot_noisy_but_close` passes vacuously.
2. **Seed reuse creates a fake gradient.** Each row of a parameter batch receives a fixed
   seed, so identical parameters give different energies per row with the *same* pattern on
   every call (spread 0.22 over five identical rows). With `fd_eps = 0.1` this is a constant
   fake gradient of about 2, which ADAM follows: it explains the RVB + ADAM climb from -14.1 to
   -10.8. The same ADAM settings converge to -17.99 without noise. SPSA is unaffected because
   its random directions average the bias away.
3. **"Best energy" is biased low under shot noise.** Taking the minimum over a noisy trace
   selects favourable fluctuations. Reports should re-measure the final parameters with a
   fixed, large shot budget.

### 1.4 The benchmark instance is easy

The open 12-site kagome star has an exact dimer-product ground state: six singlets, which a
6-CX circuit prepares. With the default Hamiltonian-cycle site order, both ansatzes reach it
at all-zero parameters. Consequences:

- Noiseless runs mostly test whether the optimiser returns to zero.
- A good compressor should rediscover the 6-CX circuit, which makes compression look better
  here than it will on real problems.
- Subspace methods are trivially exact: the S_z = 0 sector has only 924 basis states.

Every direction below should therefore be validated on at least one harder instance (M3).

## 2. Improvement directions

### 2.1 Parameter-shift gradients

*Lab note 1, written there as "phase shift".*

For a gate `exp(-i theta G)` whose generator `G` has exactly two eigenvalues `+-1/2`, the
derivative of any expectation value is exact:

```
dE/dtheta = [ E(theta + pi/2) - E(theta - pi/2) ] / 2
```

Both ansatzes qualify, gate by gate:

- **RVB:** each eSWAP is `exp(-i phi/2 SWAP)`. SWAP has eigenvalues `+-1`, so the generator
  `SWAP/2` has `+-1/2`. Each of the 12 parameters appears once, giving 24 energies per gradient.
- **HVA:** each parameter is shared by 18 rotations (`RZZ`, `RYY`, `RXX` on 6 bonds). The rule
  applies per occurrence and the occurrences are summed: 36 occurrences, 72 energies per
  gradient. A single shift of the shared parameter does *not* work, because the summed
  generator has many eigenvalues. A generalised shift rule could handle it with fewer terms.
- **Legacy eSWAP** contains a controlled rotation, whose generator has three eigenvalues. It
  needs a four-term rule; skip it.

Why it is worth trying, despite the higher count, is noise. With shot noise σ per energy:

| gradient | energies per iteration | noise per component |
|---|---|---|
| forward difference, `fd_eps = 0.1` | n + 1 (13 RVB, 3 HVA) | about 14σ |
| parameter shift, RVB | 24 | about 0.7σ |
| parameter shift, HVA | 72 | about 3σ |

For RVB this is a clear win. For HVA, finite differences given the same total shot budget
reach about 3σ too, and SPSA stays cheapest.

Implementation sketch:

- Give `RingAnsatz` an "unshared" build mode returning a circuit with one parameter per gate
  occurrence, plus the map from occurrences to the original parameters.
- Add a gradient strategy in `src/kagome_vqe/optimizers/` that builds all `2K` shifted rows,
  evaluates them as one batch, and sums per original parameter. Wire it to
  `OptimizerConfig.gradient = "parameter_shift"` for ADAM and plain gradient descent.
- Add an SPSA-gradient + ADAM-moments hybrid while there: two energies per step, ADAM's
  smoothing.

### 2.2 Circuit-depth compression

*Lab note 2.* The bias grows with the two-qubit gate count, so cutting CX gates is the most
direct way to raise the noisy floor. The lab framed two mindsets.

**(a) Recompile every step.** Compress the bound ansatz `U(theta_k)` classically, run the
compressed circuit, update `theta`, compress again.

- *Cost.* Every energy the optimiser asks for needs its own compressed circuit: 2 per SPSA
  step, n + 1 per finite-difference step, 2K per parameter-shift step.
- *Paradox to keep in mind.* A classical compressor needs a classical description of the
  target state. If that description is good enough to compress, it is usually good enough to
  compute the energy directly. At 12 qubits everything is classically tractable, so (a) is a
  research tool here, not a scalable algorithm.
- *Where (a) does scale.* Compress only a classically tractable *prefix* of the circuit, for
  example the initial state and the first layers while entanglement is low, and keep the
  remaining parameterised layers on hardware. This is the use case the tensor-network AQC
  method was designed for.
- *Overhead reducers.* Warm-start each compression from the previous solution, since `theta`
  moves little per step. Recompress only every few iterations, and gate staleness with a
  fidelity threshold.

**(b) Learn the map from original to compressed parameters.** Treat the compression as a
function `phi*(theta) = argmax_phi F(phi; theta)`, where `F` is the fidelity between the
compressed circuit `V(phi)` and `U(theta)`. Advice:

- *Implicit differentiation makes (b) concrete.* At the optimum, `grad_phi F = 0`. Then

  ```
  dphi*/dtheta = - [ d2F/dphi2 ]^(-1)  d2F/(dphi dtheta)
  ```

  Both Hessian blocks come from autodiff on a statevector or MPS simulator, using JAX or
  PyTorch. Two uses follow:
  - Cheap updates: `phi(theta + delta) ~= phi*(theta) + J delta`, with periodic
    re-compression to stop drift.
  - Gradients: `dE/dtheta = J^T dE/dphi`, where `dE/dphi` is measured on hardware by parameter
    shift on the *shallow* circuit. Directions 2.1 and 2.2 then reinforce each other.
- *Simplest version first: compress once, then train the compressed circuit directly.*
  Compress the ansatz at a good point, such as the noiseless optimum, and use `V(phi)` as a
  warm-started ansatz whose parameters are optimised on hardware. The original parameters
  `theta` drop out.
- *Keep the physics in the template.* A generic compressed template loses the singlet
  structure and invites barren plateaus. Build the template from spin-rotation-invariant
  two-qubit gates, `exp(-i a (XX + YY + ZZ))`, the same family as eSWAP. The compressed circuit
  then stays in the total-spin sector of the ground state, and the warm start sits near the
  answer.

**Recommendation:** start with the compress-once variant of (b), then add the implicit
Jacobian, and run (a) only as an upper-bound reference on 12 qubits.

**Compression methods beyond AQC.** Several families are worth comparing:

| family | tool | what it compresses | notes |
|---|---|---|---|
| peephole and commutation rules, exact | Qiskit preset level 3 | gate sequences | already in use |
| approximate two-qubit resynthesis | Qiskit `approximation_degree` | each two-qubit block | one-line experiment |
| ZX-calculus rewriting, exact | `pyzx` | whole circuit | strong on Clifford-heavy parts |
| partition and numerical resynthesis | `bqskit` | blocks of 3 to 4 qubits | approximate with a threshold |
| Pauli-network synthesis | Qiskit `synth_pauli_network_rustiq`, HLS plugin `PauliEvolution.rustiq` | products of Pauli exponentials | fits HVA if its layers are built as `PauliEvolutionGate` |
| unitary AQC | `qiskit.synthesis.unitary.aqc.AQC` | full unitary | cost grows as 4^n; fine near 12 qubits |
| state AQC with matrix product states | `qiskit-addon-aqc-tensor` | prepared state only | all VQE needs; designed for Trotter circuits |
| variational compiling on hardware | own code | fidelity measured on the device | quantum-assisted quantum compiling |
| operator backpropagation | `qiskit-addon-obp` | trailing layers, per observable | complementary: shrinks the circuit by moving work into the observable |

VQE only needs the state prepared from `|0...0>`, not the full unitary. State-based methods
such as MPS AQC are much cheaper than unitary AQC for that reason.

### 2.3 Error mitigation

Mitigation estimates and removes the remaining bias, and it composes with everything else.

- **Zero-noise extrapolation:** amplify noise by gate folding, measure at several noise
  levels, extrapolate to zero. On hardware this is `resilience_level = 2`. The runtime's local
  mode ignores resilience levels, so simulator studies need client-side folding.
- **Twin-circuit rescaling:** run a circuit with the same gate structure whose ideal energy is
  classically known, measure its damping factor, and divide it out of the real measurement.
  This relies on the near-depolarising picture of section 1.2. One option is the mirror
  circuit `U(theta) U(theta)^dagger` on the singlet product, whose ideal energy is -18. It has
  twice the depth, so take the square root of its damping factor.
- **Readout mitigation:** TREX via `resilience_level = 1` on hardware, or `mthree` on sampled
  counts. This only matters once readout error is modelled, after M0.
- **Pauli twirling and dynamical decoupling:** both are runtime options. Twirling turns
  coherent errors into stochastic ones, which makes rescaling and extrapolation more reliable.
- **Symmetry filtering:** the Hamiltonian conserves total S_z, and the ground state has
  S_z = 0. Shots measured in the ZZ basis that violate it can be discarded. This needs sampled
  shots, so it also depends on M0.

### 2.4 Subspace methods beyond VQE: QSE, KQD, SQD, SKQD

*Lab note 3.* Integrating Krylov quantum diagonalisation (KQD) was raised, with the practical
path left open. Advice follows.

**Why subspace methods help with bias.** They do not report the energy of one noisy state.
They measure matrix elements of `H` and the overlap matrix `S` in a small subspace, then
diagonalise the projected problem classically. The sample-based variants (SQD, SKQD) go
further. They use the quantum device only to *choose* a set of computational basis states,
then compute the energy exactly within that set. The result is a variational upper bound
whose value is not damped by depolarising noise; noise only degrades the choice of subspace.

**Four methods, from cheapest to most ambitious:**

| method | subspace | new quantum circuits | main risk |
|---|---|---|---|
| QSE, quantum subspace expansion | `{psi, P_a psi}` for a few Pauli or spin-flip operators `P_a` around the VQE state | none: only extra Pauli expectation values | operator count grows quadratically |
| KQD, Krylov | `{psi, e^(-iH dt) psi, e^(-2iH dt) psi, ...}` | Trotterised time evolution | deep circuits; noisy overlaps make the eigenproblem ill-conditioned |
| SQD, sample-based | basis states sampled from one circuit | none beyond sampling | Heisenberg ground states are not sparse in the Z basis |
| SKQD, sample-based Krylov | basis states sampled from Krylov states | Trotterised time evolution | combines the depth of KQD with the sparsity question of SQD |

**Recommended path:**

1. **QSE first.** It reuses the VQE circuit exactly. Every matrix element
   `<psi| P_a H P_b |psi>` is a sum of Pauli expectation values, so the existing estimator
   layer computes it. Use spin-flip pairs `S+_i S-_j` on bonds, which preserve S_z, as the
   expansion operators. QSE is also known to correct part of the decoherence error.
2. **KQD in exact simulation**, before any device time. Study the Krylov dimension, the time
   step `dt` (a common starting point is a fraction of `pi / ||H||`), and the threshold for
   discarding small eigenvalues of `S`. This is cheap on 12 qubits and fixes the hyper-
   parameters.
3. **KQD with the conservation trick on noisy simulation.** Overlaps normally need a
   Hadamard test with a *controlled* time evolution, which is too deep. The Heisenberg model
   conserves total S_z, and the all-up state `|0...0>` is an exact eigenstate with energy +18.
   Preparing a superposition of the reference state with `|0...0>`, whose evolution is just
   a known phase, lets the overlaps be read from ordinary measurements, with no controlled
   evolution. IBM's Krylov experiments on lattices up to 56 sites used conservation laws in
   this spirit to avoid controlled evolutions (Yoshioka et al.); check the paper for the exact
   circuit before implementing.
4. **Compress the Krylov circuits.** Trotter circuits are exactly what MPS AQC was designed
   for, so direction 2.2 feeds directly into this step.
5. **SKQD and SQD last**, and only on larger lattices. At 12 sites the sampled subspace
   quickly becomes the whole 924-state sector, which is trivially exact. On larger frustrated
   lattices the open question is sparsity. A possible research angle is sampling in a rotated
   basis where singlet-dominated states are sparse.

**Integration with VQE.** The VQE state becomes the *reference* for the subspace method. It
does not need to be accurate, only to overlap well with the ground state, which shrinks the
Krylov dimension needed. Store both energies in `result.json`: the raw VQE value and the
subspace-corrected value. A future `kagome_vqe.subspace` module would hold `qse_energy`,
`krylov_energy` and the sampling variants, all built on the existing estimator protocol.

### 2.5 Optimiser hygiene

These matter only once the bias is reduced, but they are cheap.

- Prefer SPSA, or calibrated SPSA, on noisy backends. Use the SPSA-ADAM hybrid from 2.1.
- Average a few gradient estimates per step, or resample, when the noise floor dominates.
- Increase shots as the optimisation converges.
- Report the final energy re-measured with a fixed large shot budget, never the trace minimum.

## 3. Plan

Each milestone has a benchmark check that must pass before the next builds on it. Hardware
checkpoints are marked; the rest runs on simulators.

### M0. Honest noisy simulation

- **Deliverables:**
  - Switch `aer_ideal` and `aer_noisy` to a sampling estimator, for example `qiskit.primitives.BackendEstimatorV2` over `AerSimulator`.
  - Draw a fresh seed per call from the run's generator.
  - Add a final re-measurement step to `run_vqe`, with `final_shots` in the config.
  - Fix the vacuous Aer unit test.
- **Benchmark check:**
  - Standard error falls as 1/√shots.
  - Identical rows differ between calls.
  - Readout error alone moves the energy away from -18.
  - Re-run `benchmark_fake_guadalupe.toml`; RVB + ADAM no longer climbs.
- **Exit:** a new noisy baseline table in the README, replacing the current one.
- **Hardware checkpoint H0:** one HVA + SPSA run on a current IBM device using
  `configs/ibm_hva_spsa.toml`, to calibrate expectations against the fake backend.

### M1. Better gradients

- **Deliverables:**
  - Unshared-parameter build mode.
  - Parameter-shift gradient.
  - SPSA-ADAM hybrid.
  - Unit tests: parameter-shift gradients match finite differences noiselessly; evaluation counts are 24 and 72.
- **Benchmark check:** at an *equal total shot budget*, compare finite differences, parameter
  shift, SPSA and the hybrid on RVB and HVA. Plot best-so-far energy against shots spent.
- **Exit:** a recommended default gradient per ansatz, recorded in the configs.

### M2. Error-mitigation baseline

- **Deliverables:**
  - Client-side ZNE by gate folding.
  - Twin-circuit rescaling.
  - S_z symmetry filtering.
  - Readout mitigation with `mthree`.
  - Each wrapped as an estimator decorator so it composes with any backend kind.
- **Benchmark check:**
  - Energy at the exact parameters on FakeGuadalupeV2, per method and per combination.
  - Gap closed and variance cost reported against the unmitigated -13.97.
- **Exit:** a mitigation stack chosen for hardware.
- **Hardware checkpoint H1:** the same check on a device with runtime resilience levels 1 and 2.

### M3. Harder benchmark instances

- **Deliverables:**
  - `kagome_lattice(rows, cols, periodic=...)` with a matching site order.
  - Exact references in symmetry sectors, up to roughly 20 to 24 sites.
  - The identity site order kept as an in-place hard mode.
- **Benchmark check:**
  - Ground-state energies match published or independently computed values.
  - The ring ansatzes no longer start at the exact answer.
- **Exit:** at least one instance where noiseless VQE has a visible gap, used by M4 and M5.

### M4. Depth compression

- **Deliverables:**
  - **Step 1:** transpiler-level experiments: `approximation_degree`, PyZX, BQSKit, Rustiq for HVA built from Pauli evolutions.
  - **Step 2:** compress-once-then-train using a spin-rotation-invariant template.
  - **Step 3:** implicit-Jacobian updates and chain-rule gradients.
  - **Step 4:** per-step recompilation (mindset a) as a 12-qubit reference only.
- **Benchmark check:**
  - For each method: CX count, compression fidelity, noisy energy at fixed parameters, and full noisy VQE energy.
  - On the 12-site star, the compressor should approach the 6-CX singlet circuit; treat that as a sanity check, not a result.
- **Exit:** a compression pipeline that raises the noisy floor on an M3 instance.
- **Hardware checkpoint H2:** best compressed circuit plus the M2 mitigation stack on a device.

### M5. Subspace methods

- **Deliverables:** `kagome_vqe.subspace` with QSE first, then KQD in exact simulation, then
  KQD with the S_z trick on noisy simulation, then compressed Krylov circuits.
  SKQD and SQD come last, on M3 instances only.
- **Benchmark check:**
  - Subspace-corrected energy against raw VQE at an equal shot budget.
  - Conditioning of `S` against noise level.
  - Krylov dimension needed against reference quality.
- **Exit:** a decision on whether the subspace step justifies its shot cost for the target
  lattice size.
- **Hardware checkpoint H3:** QSE on device, reusing the H2 circuit.

### Summary

| milestone | depends on | main question |
|---|---|---|
| M0 honest simulation | none | are the noisy numbers real? |
| M1 gradients | M0 | which gradient gives the most precision per shot? |
| M2 mitigation | M0 | how much of the bias can be removed after the fact? |
| M3 harder instances | none, can run in parallel | is the benchmark testing anything? |
| M4 compression | M2, M3 | how far can depth fall before fidelity does? |
| M5 subspace methods | M2, M3, and M4 for Krylov | can classical post-processing remove the rest? |

## 4. Benchmark protocol

Every benchmark that feeds a decision above reports, per run:

- Exact reference energy and how it was obtained.
- Final energy re-measured with a fixed shot budget, with its standard error.
- Two-qubit gate count and depth after transpilation, and the physical layout.
- Total shots and wall-clock or device time.
- Backend name, noise model or calibration date, mitigation settings.
- Seeds and the full config file.

The results directory already stores most of this. M0 adds the re-measured energy and total
shots.

## 5. References

- K. Mitarai et al., Phys. Rev. A 98, 032309 (2018); M. Schuld et al., Phys. Rev. A 99,
  032331 (2019): parameter-shift rules.
- D. Wierichs et al., Quantum 6, 677 (2022): general parameter-shift rules.
- L. Madden and A. Simonetto, "Best approximate quantum compiling problems",
  arXiv:2106.05649: the unitary AQC behind `qiskit.synthesis.unitary.aqc`.
- N. F. Robertson et al., "Approximate Quantum Compiling for Quantum Simulation: A Tensor
  Network based approach", arXiv:2301.08609: the method behind `qiskit-addon-aqc-tensor`.
- S. Khatri et al., "Quantum-assisted quantum compiling", Quantum 3, 140 (2019).
- K. Temme, S. Bravyi and J. M. Gambetta, Phys. Rev. Lett. 119, 180509 (2017): zero-noise
  extrapolation.
- J. R. McClean et al., Phys. Rev. A 95, 042308 (2017): quantum subspace expansion.
- N. Yoshioka et al., "Diagonalization of large many-body Hamiltonians on a quantum
  processor", arXiv:2407.14431, Nature Communications (2025): Krylov diagonalisation on
  lattices up to 56 sites.
- J. Robledo-Moreno et al., "Chemistry Beyond Exact Solutions on a Quantum-Centric
  Supercomputer", arXiv:2405.05068, Science Advances (2025): sample-based quantum
  diagonalisation.
- J. Yu et al., "Quantum-Centric Algorithm for Sample-Based Krylov Diagonalization",
  arXiv:2501.09702 (2025).
