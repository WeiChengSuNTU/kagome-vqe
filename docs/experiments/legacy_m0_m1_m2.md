# Legacy experiments: M0, M1 and M2 on the 12-site kagome star

These are the experiments of roadmap milestones M0 (honest noisy simulation), M1 (gradients)
and M2 (error mitigation), all on the **open 12-site kagome star** with the **legacy ring
ansatzes** (`hva`: 2 shared parameters; `rvb`: 12). The sections below were moved here
unchanged from the README (November 2026) when the README was reorganised; numbers, tables
and conclusions are exactly as they were. The diagnosis and plan behind them are in
[`../roadmap.md`](../roadmap.md); the milestone report and deck in `report/` and `slides/` are
generated from the same `results/` directories by `docsgen`.

> **Scope of the M1 optimiser recommendation.** "HVA: use plain SPSA" and "RVB: use ADAM with
> parameter shift" were established for the legacy low-parameter ansatzes: HVA with **two**
> shared parameters and RVB with twelve, on the star, whose ground state is a dimer product.
> They do **not** carry over automatically to the generalised graph HVA with one parameter
> per gate (`graph_hva`, `N_theta = p * N_b`, e.g. 16 to 64 parameters on the 2x6 rectangle).
> That question is studied separately in [`graph_hva_optimizers.md`](graph_hva_optimizers.md).

Hamiltonian convention throughout: `H = sum over the 18 bonds of (XX + YY + ZZ)` (Pauli
normalisation, `= 4 sum S.S`), exact ground energy -18.

## Conventions

Every number below is the optimiser's final parameters re-measured once, not the lowest value
seen during optimisation, which is biased low under shot noise.

## Exact statevector

Seed 42, the paper's hyperparameters (`maxiter = 100`, `lr = 0.01`, `perturbation = 0.01`),
initial point `x0 = pi/4 * rand`. Produced by `kagome-vqe benchmark`.

| run | final E (re-measured) | exact E | rel. err | 2q gates | evaluations | total shots | time (s) |
|---|---|---|---|---|---|---|---|
| rvb_spsa | -17.8369 | -18.0000 | 0.906 % | 78 | 201 | 0 | 1.7 |
| rvb_adam | -17.9980 | -18.0000 | 0.011 % | 78 | 1301 | 0 | 11.2 |
| hva_spsa | -18.0000 | -18.0000 | 0.000 % | 78 | 201 | 0 | 2.2 |
| hva_adam | -18.0000 | -18.0000 | 0.000 % | 78 | 301 | 0 | 3.3 |

Evaluation counts follow the optimisers' cost model: ADAM with forward differences spends
`n + 1` evaluations per iteration (13 for RVB, 3 for HVA) plus one final evaluation; SPSA
spends 2 per iteration plus one final evaluation. The two-qubit gate count is for the logical
circuit; nothing is transpiled for hardware here.

## Noisy FakeGuadalupeV2

Produced by `kagome-vqe benchmark configs/benchmark_fake_guadalupe.toml`: `aer_noisy` on
`FakeGuadalupeV2`, heavy-hex loop layout, seed 42, 4096 shots per measurement basis, final
re-measurement with 16384 shots, `x0 = 0.3 * rand`. SPSA runs 50 iterations
(`lr = 0.05`, `perturbation = 0.1`); ADAM runs 30 (`lr = 0.05`, `fd_eps = 0.1`).

| run | final E (re-measured) | exact E | rel. err | 2q gates | evaluations | total shots | time (s) |
|---|---|---|---|---|---|---|---|
| rvb_spsa | -12.7168 | -18.0000 | 29.351 % | 66 | 101 | 1,290,240 | 266.9 |
| rvb_adam | -12.7188 | -18.0000 | 29.340 % | 66 | 391 | 4,853,760 | 1040.3 |
| hva_spsa | -12.8689 | -18.0000 | 28.506 % | 66 | 101 | 1,290,240 | 273.2 |
| hva_adam | -12.6729 | -18.0000 | 29.595 % | 66 | 91 | 1,167,360 | 244.2 |

All four combinations end at the noise floor. Measured directly at the exact ground-state
parameters, with 16384 shots:

| ansatz | CX gates | readout error only | gate + T1/T2 error only | full noise |
|---|---|---|---|---|
| HVA | 66 | -16.43 | -14.01 | -12.78 |
| RVB | 66 | -16.43 | -13.94 | -12.70 |

The two error sources combine multiplicatively: readout keeps 91 % of the ideal energy, gate
errors keep 78 %, and their product predicts the full-noise value. No optimiser setting can
get below this floor; closing the gap is the job of error mitigation and shallower circuits,
as planned in the roadmap. Results before October 2026 used an Aer estimator that never
sampled measurements; they omitted readout error and reported a floor near -14.

## Gradient comparison (roadmap M1)

Produced by `configs/m1_gradients_ideal.toml` and `configs/m1_gradients_noisy.toml`. Every
method spends the same 24576 shots per measurement basis per iteration, split over the
energies its gradient needs, for 40 iterations without gate noise and 20 with it. All gradient
methods share ADAM with `lr = 0.05`; plain SPSA uses its own update with `lr = 0.01`. The
metric is the noise-free energy of the final parameters, relative to -18: how good the
parameters are, separately from the noise floor. The noisy run covers RVB only, because noisy
simulation costs about 2.5 s per energy and HVA parameter shift needs 72 energies per step.

| ansatz | method | energies per iteration | noise-free error, shot noise only (mean ± sample std, 5 seeds) | noise-free error, FakeGuadalupeV2 noise (1 seed) |
|---|---|---|---|---|
| HVA | ADAM, forward differences (the paper) | 3 | 0.252 % ± 0.218 | not run |
| HVA | ADAM, central differences | 5 | 0.125 % ± 0.142 | not run |
| HVA | ADAM, SPSA gradient (hybrid) | 2 | 0.113 % ± 0.137 | not run |
| HVA | ADAM, parameter shift | 72 | 0.126 % ± 0.107 | not run |
| HVA | plain SPSA | 2 | 0.001 % ± 0.001 | not run |
| RVB | ADAM, forward differences (the paper) | 13 | 0.149 % ± 0.111 | 0.503 % |
| RVB | ADAM, central differences | 25 | 0.039 % ± 0.015 | not run |
| RVB | ADAM, SPSA gradient (hybrid) | 2 | 0.172 % ± 0.179 | 1.428 % |
| RVB | ADAM, parameter shift | 24 | 0.022 % ± 0.012 | 0.079 % |
| RVB | plain SPSA | 2 | 1.756 % ± 0.573 | 3.787 % |

- **RVB: use ADAM with parameter shift.** It gives the most precise parameters at the same
  shot budget, without and with noise. Plain SPSA is the least precise for RVB's 12 parameters.
- **HVA: use plain SPSA.** With only two parameters it reaches the most precise parameters,
  while parameter shift costs 72 energies per step. The ADAM variants all plateau at a similar
  level, which points to ADAM's fixed step size as the limit rather than the gradients.
- `configs/fake_guadalupe_rvb_adam_ps.toml` and `configs/ibm_hva_spsa.toml` encode these
  choices. The noise-free errors per shot spent are plotted in
  `results/m1_ideal-summary/ideal_vs_shots.png`.

## Error mitigation (roadmap M2)

Produced by `kagome-vqe benchmark configs/m2_mitigation.toml --point 0 --repeats 8` and
`--point 0.3`, plus `--set backend.noise=readout` and `--set backend.noise=gates` to apply one
error source at a time. Each cell is the relative bias of the mean against the noise-free energy,
in %, with its standard error, on `FakeGuadalupeV2` at 4096 shots per basis. Every repeat builds
its own estimator, so one-off measurements (readout calibration, twin) are redone and their
noise is part of the error. ⚠ marks a mean significantly below -18, which by the variational
principle is an artefact.

| stack | full noise, exact params | full noise, θ = 0.3 | readout noise only, θ = 0.3 | gate noise only, θ = 0.3 | shots per energy | one-off shots |
|---|---|---|---|---|---|---|
| none (HVA) | +29.08 ± 0.12 | +28.40 ± 0.15 | +8.17 ± 0.16 | +21.96 ± 0.15 | 12,288 | 0 |
| none (RVB) | +29.27 ± 0.09 | +29.21 ± 0.19 | +8.45 ± 0.13 | +22.66 ± 0.13 | 12,288 | 0 |
| readout (HVA) | +22.11 ± 0.18 | +22.11 ± 0.19 | +0.12 ± 0.13 | +22.07 ± 0.15 | 12,288 | 32,768 |
| readout (RVB) | +22.50 ± 0.17 | +22.24 ± 0.19 | -0.03 ± 0.09 | +22.28 ± 0.15 | 12,288 | 32,768 |
| symmetry (HVA) | +8.64 ± 0.09 | +8.01 ± 0.15 | +0.60 ± 0.11 | +4.46 ± 0.16 | 12,288 | 0 |
| symmetry (RVB) | +8.97 ± 0.13 | +8.84 ± 0.14 | +0.74 ± 0.05 | +4.96 ± 0.11 | 12,288 | 0 |
| readout+symmetry (HVA) | +5.10 ± 0.10 | +4.38 ± 0.22 | -0.01 ± 0.12 | +4.37 ± 0.16 | 12,288 | 32,768 |
| readout+symmetry (RVB) | +5.24 ± 0.14 | +4.90 ± 0.24 | -0.03 ± 0.05 | +4.96 ± 0.11 | 12,288 | 32,768 |
| zne (HVA) | +10.79 ± 0.21 | +9.05 ± 0.20 | +8.17 ± 0.16 | +0.96 ± 0.19 | 36,864 | 0 |
| zne (RVB) | +9.87 ± 0.20 | +9.99 ± 0.24 | +8.45 ± 0.13 | +1.63 ± 0.18 | 36,864 | 0 |
| readout+zne (HVA) | +1.81 ± 0.31 | +1.60 ± 0.25 | +0.12 ± 0.13 | +1.57 ± 0.17 | 36,864 | 98,304 |
| readout+zne (RVB) | +1.44 ± 0.19 | +1.07 ± 0.30 | -0.03 ± 0.09 | +1.13 ± 0.23 | 36,864 | 98,304 |
| readout+symmetry+zne (HVA) | -7.33 ± 0.28 ⚠ | -7.27 ± 0.25 | -0.01 ± 0.12 | -7.32 ± 0.22 | 36,864 | 98,304 |
| readout+symmetry+zne (RVB) | -7.69 ± 0.20 ⚠ | -7.79 ± 0.38 ⚠ | -0.03 ± 0.05 | -7.59 ± 0.17 ⚠ | 36,864 | 98,304 |
| twin (HVA) | -4.77 ± 0.47 ⚠ | -5.48 ± 0.39 | -0.58 ± 0.29 | -4.82 ± 0.37 | 12,288 | 49,152 |
| twin (RVB) | -0.61 ± 0.18 ⚠ | -0.91 ± 0.23 | +0.59 ± 0.12 | -1.41 ± 0.25 | 12,288 | 49,152 |
| readout+twin (HVA) | -4.78 ± 0.20 ⚠ | -4.62 ± 0.32 | -0.19 ± 0.22 | -4.83 ± 0.32 | 12,288 | 81,920 |
| readout+twin (RVB) | -1.23 ± 0.24 ⚠ | -1.36 ± 0.18 | +0.05 ± 0.18 | -1.40 ± 0.21 | 12,288 | 81,920 |
| **readout+symmetry+twin** (HVA) | +1.94 ± 0.25 | +1.06 ± 0.31 | -0.23 ± 0.22 | +0.63 ± 0.24 | 12,288 | 81,920 |
| **readout+symmetry+twin** (RVB) | +0.53 ± 0.17 | +0.35 ± 0.18 | -0.08 ± 0.06 | +0.56 ± 0.12 | 12,288 | 81,920 |

- **Chosen stack: readout correction, then symmetry filtering, then twin rescaling.**
  - Its bias is +0.4 to +1.9 % under full noise, +0.6 % with gate noise alone and about 0 with
    readout noise alone.
  - The documentation worktree's exact density-matrix analysis, with no sampling, gives an
    infinite-shot limit of +0.5 to +1.3 %, consistent with these numbers.
  - The bias is positive everywhere: the stack under-corrects slightly and never overshoots.
  - It works on each error source separately, so no errors are cancelling.
  - The highest value, +1.94 % for HVA at the exact parameters, was a fluctuation in the twin
    measurement. Two reruns with new seeds and 16 repeats give +1.32 ± 0.22 % and
    +1.14 ± 0.14 %, matching the +1.27 % limit.
  - It costs no extra shots per energy, plus 81,920 one-off shots.
- **Readout correction plus ZNE is comparable for HVA,** at +1.6 to +1.8 %, but costs three
  times the shots per energy and is worse for RVB. It remains the fallback for ansatzes without
  the singlet symmetry.
- **Why the twin works after filtering.** The filter leaves only noise inside the singlet sector,
  which damps different states much more uniformly. The documentation worktree measured the
  mismatch in damping between HVA states falling from 4.7 % to 1.3 %: much more uniform, though
  not exactly. The remaining mismatch is the residual bias.
- **Readout correction plus symmetry filtering alone leaves a 4 to 5 % gate-noise residual**
  that the filter cannot see.
- **Never combine symmetry filtering with ZNE.** It over-corrects by about 7 % even with gate
  noise alone.
- **Limits.** Correct-then-filter works on the full outcome distribution and the automatic twin
  needs a classical simulation. Both cap at about 20 qubits and need replacing before larger
  lattices (M3).

The chosen stack in a full optimisation (`configs/m2_mitigated_vqe.toml`), with the
M1-recommended optimisers, on full `FakeGuadalupeV2` noise:

| run | final energy, re-measured | relative error | noise-free energy of final parameters | total shots |
|---|---|---|---|---|
| HVA, plain SPSA | -17.958 ± 0.116 | 0.23 % | -17.995 | 1,372,160 |
| RVB, ADAM with parameter shift | -17.898 ± 0.068 | 0.57 % | -17.961 | 1,608,704 |

Unmitigated, these runs end near -12.7 (29 % error). Neither reading lies below -18; the
error bars include the twin's uncertainty, since a single run measures its twin once.

**Correction history.** The M2 result was corrected twice, both times after the documentation
worktree recomputed it exactly.

1. **Commit `c9272e3`** recommended readout plus symmetry, applied filter-first. With readout
   noise alone it over-corrected by about 9 % (-19.57 instead of -18). Under full noise that
   happened to cancel the gate-noise residual on this device. Commit `56aadd3` switched to
   correct-then-filter and added the noise-source switch.
2. **The follow-up study measured the readout calibration and the twin once** and shared them
   across repeats, so its spread hid their noise and it claimed a bias of at most 0.43 %.
   Commit `2116da4` redoes them in every repeat; the tables above come from that run.

Earlier results remain in `results/m2-*` and `results/m2b-*`.
