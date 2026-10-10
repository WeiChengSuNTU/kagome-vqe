# Roadmap: closing the noisy-energy gap

Status: proposal, October 2026. Owner: the kagome-vqe group.

This document records what limits the precision of noisy VQE runs today, the improvement
directions discussed in the lab, advice on each, and an ordered plan with a benchmark check
for every step. It is meant to be updated as milestones land: tick the exit criteria, link the
results directories, and move findings into `docs/physics.md` once they are settled.

## 1. Where we are

### 1.1 Measured baseline

Updated after M0. Energies are the final parameters re-measured once.

| setting | final energy | relative error | source |
|---|---|---|---|
| exact statevector, HVA + ADAM | -18.0000 | 0.000 % | `configs/benchmark.toml` |
| exact statevector, RVB + SPSA | -17.8369 | 0.906 % | `configs/benchmark.toml` |
| noisy FakeGuadalupeV2, all four combos | -12.67 to -12.87 | 28.5 to 29.6 % | `configs/benchmark_fake_guadalupe.toml` |

Full tables are in the README. Before M0 the noisy figure was -14.0 to -14.1; that simulation
never sampled measurements, so it omitted readout error.

### 1.2 Diagnosis: the noisy runs converged to a biased floor

The optimiser is not the bottleneck. Evaluating the circuits at the *exact* ground-state
parameters on the same noise model gives the same plateau. Sampled with 16384 shots, after M0:

| ansatz | CX gates | noise-free | readout error only | gate + T1/T2 only | full noise |
|---|---|---|---|---|---|
| HVA | 66 | -18.000 | -16.43 | -14.01 | -12.78 |
| RVB | 66 | -18.000 | -16.43 | -13.94 | -12.70 |

Noise turns the prepared state into a mixture. Under a depolarising model,
`E_measured ~= (1 - p) * E_ideal`, because every Pauli term averages to zero on the maximally
mixed state. The factors multiply: readout keeps 91 % of the energy, gate and decoherence
errors keep 78 %, and `18 * 0.913 * 0.778 = 12.79` matches the full-noise value. That
near-multiplicative structure is what makes rescaling-type mitigation (section 2.3) promising.
More iterations cannot remove this: it is a bias, not a convergence failure. The variational
principle also holds for mixed states, so a noisy reading below -18 signals an artefact.

### 1.3 Known defects in the noisy simulation (fixed in M0)

These were found during the diagnosis. All three are fixed by milestone M0; see section 3.

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

### M0. Honest noisy simulation (done)

**Status, October 2026.** Code in commit `0433fd7`; results in
`results/statevector_m0-summary` and `results/fake_guadalupe_m0-summary`.

| check | result |
|---|---|
| standard error falls as 1/√shots | real spread over 40 repeats of HVA at the exact parameters with FakeGuadalupeV2 noise: 0.144 at 1024 shots, 0.060 at 4096; unit-tested. Without noise the spread at the exact parameters is exactly zero, because each triangle's unmatched bonds cancel shot by shot |
| identical rows differ between calls | yes, while the same run seed reproduces the sequence; unit-tested |
| readout error alone moves the energy | -18.00 to -16.43 at the exact parameters; unit-tested |
| RVB + ADAM no longer climbs | final -12.72, at the floor; previously drifted to -10.8 |
| new noisy baseline in the README | done |
| H0 hardware run | skipped for now at the group's request; the IBM code path is kept and tested in local mode |

Two findings to carry forward:

- The standard error reported by the sampling primitive is the bound
  `sum_i |c_i| sigma_i / sqrt(shots)`, about ten times the real spread for this Hamiltonian.
  Use repeated measurements, not the reported `std`, when comparing methods in M1 and M2.
- Noisy density-matrix simulation costs about 2.5 s per energy whatever the shot count, so
  simulator benchmarks can use large shot budgets for free. Shots are not free on hardware.

Original plan:


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

### M1. Better gradients (done)

**Status, October 2026.** Code in commit `1a27c28`; results in `results/m1_ideal-summary` and
`results/m1_noisy-summary`; the full table is in the README under "Gradient comparison".

| check | result |
|---|---|
| parameter shift matches finite differences without noise | to about 1e-9, for HVA, RVB and `real_amplitudes`; unit-tested |
| energies per gradient | 72 for HVA, 24 for RVB; unit-tested |
| unsharing changes the circuit | no: same two-qubit gate count (66 on Guadalupe) |
| equal-budget comparison | 5 seeds without gate noise, 1 seed on FakeGuadalupeV2 noise (RVB) |
| recommended default per ansatz | RVB: ADAM with parameter shift. HVA: plain SPSA |

Measured at equal shots per iteration, as noise-free error of the final parameters:

- **RVB, ADAM with parameter shift:** 0.022 % with shot noise only, 0.079 % on noisy Guadalupe.
- **RVB, the paper's forward differences:** 0.15 % and 0.50 % in the same two settings.
- **HVA, plain SPSA:** 0.001 %, against 0.11 to 0.25 % for every ADAM variant.

Follow-ups this milestone uncovered:

- **ADAM's step size, not its gradient, limits HVA.** With a fixed `lr = 0.05` every ADAM
  variant plateaus near 0.1 %. A decaying learning rate should be tried before concluding
  that SPSA is intrinsically better for HVA.
- **RVB had not converged when the budget ran out.** Its ADAM curves are still falling at
  three million shots, so longer runs may reorder the close finishers.
- **The noisy check used one seed.** Repeat it with more seeds before relying on the
  smaller differences.

Original plan:


- **Deliverables:**
  - Unshared-parameter build mode.
  - Parameter-shift gradient.
  - SPSA-ADAM hybrid.
  - Unit tests: parameter-shift gradients match finite differences noiselessly; evaluation counts are 24 and 72.
- **Benchmark check:** at an *equal total shot budget*, compare finite differences, parameter
  shift, SPSA and the hybrid on RVB and HVA. Plot best-so-far energy against shots spent.
- **Exit:** a recommended default gradient per ansatz, recorded in the configs.

### M2. Error-mitigation baseline (done, revised)

**Status, October 2026.** Code in `c9272e3` (techniques) and `56aadd3` (correct-then-filter,
noise-source switch). Results in `results/m2b-evaluation-0`, `results/m2b-evaluation-0p3`,
`results/m2b_readout-evaluation-0p3`, `results/m2b_gates-evaluation-0p3` and
`results/m2b_vqe-summary`. Full tables are in the README under "Error mitigation".

| check | result |
|---|---|
| each technique implemented as a composable estimator layer | readout correction and symmetry filtering (counts level), ZNE, twin rescaling; unit-tested |
| energy at fixed points, per stack | 10 stacks x 2 ansatzes x 2 points, 8 repeats; plus readout-only and gate-only noise at θ = 0.3 |
| gap closed and variance cost | readout + symmetry + twin: bias at most 0.43 % in every noise setting, no extra shots per energy, 81,920 one-off shots |
| mitigation stack chosen for hardware | readout correction, then symmetry filtering, then twin rescaling; readout + ZNE as the fallback for non-singlet ansatzes |
| H1 hardware check | skipped for now, as with H0 |

**How the first conclusion went wrong.** Commit `c9272e3` first recommended readout plus
symmetry alone, filtering shots first and then correcting each kept shot. The documentation
worktree found, and an independent check confirmed, that this was two errors cancelling:

- **The filter already removes nearly every single readout flip.** The per-shot weights then
  divide each two-body term by about `(1 - 2e_a)(1 - 2e_b)`. With readout noise alone this gave
  -19.57 instead of -18, a 9 % over-correction.
- **The filter cannot see gate noise inside the singlet sector,** which leaves a residual of
  about 5 %.
- **On FakeGuadalupeV2 the two happened to cancel.** On a device with a different ratio of
  readout to gate error, or after compression reduces the gate error, the stack would have
  overshot.

The lesson for every later milestone: check a mitigation method against each error source
separately (`backend.noise = "readout"` / `"gates"`), not only against the full model.

Findings and open problems:

- **Correct-then-filter is exact for independent readout errors.** It inverts the calibrated
  tensored channel on the full outcome distribution, projects onto the singlet sector and
  renormalises. That caps it at about 20 qubits; larger lattices need a sector-restricted
  version.
- **Twin rescaling works after filtering, even for HVA.** The residual noise inside the singlet
  sector damps states almost uniformly, the assumption rescaling needs. The automatic twin
  needs a classical simulation (the π/4 grid is not Clifford), the same 20-qubit limit.
- **Symmetry filtering and ZNE must not be combined.** They over-correct by about 8 % even with
  gate noise alone.
- **The bootstrap standard error of correct-then-filter matches the spread between repeats,**
  within the precision 8 repeats allow (ratios 0.6 to 1.7).
- **The benchmark instance is still easy (M3).** The exact-parameter state is a product of
  singlets. The generic point and the per-source checks confirm the stack away from it, but
  harder lattices are needed before generalising.

Original plan:


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

**Early finding, October 2026** (from the documentation worktree's report, verified on
`main`). Transpiling with the parameters left free keeps each Heisenberg bond block at 5 CX,
because two-qubit resynthesis only works on numeric matrices. Binding the angles first lets the
transpiler use the 3-CX form: HVA drops from 66 to 42 CX, and at a generic point
(`phi = gamma = 0.3`) the noisy energy keeps 78 % of the ideal value instead of 71 %. Two ways to
use it: build the bond block from a parametric 3-CX template, or transpile per evaluation with
bound parameters. This is a nearly free first step for M4.


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
