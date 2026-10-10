# kagome-vqe

Variational quantum eigensolver (VQE) for the spin-1/2 Heisenberg antiferromagnet on the
12-site kagome unit cell, built on Qiskit 2.x and ready for IBM Quantum hardware.

The package rebuilds the IBM Quantum Open Science Prize 2022 kagome challenge, and the
resonating-valence-bond (RVB) and Hamiltonian-variational (HVA) ansatz comparison from
Shaffar et al. (KICS 2023), as a typed, tested, reproducible Python project. One TOML file
describes a run; one flag moves it from an exact simulator to a noisy fake device to a real
IBM Quantum system.

## What it does

The code is a pipeline whose stages map one-to-one onto modules in `src/kagome_vqe/`:

| stage | module | responsibility |
|---|---|---|
| 1 | `lattice.py` | the spin graph (sites, bonds, and which ansatz qubit hosts which site) |
| 2 | `hamiltonian.py` | the graph as a `SparsePauliOp`; exact diagonalisation for reference energies |
| 3 | `ansatz/` | parameterised trial circuits: `RVBAnsatz`, `HVAnsatz`, library wrappers |
| 4 | `transpile.py` | compile circuit + observable for a backend (`observable.apply_layout`) |
| 5 | `estimators.py` | evaluate energies on statevector, Aer, fake backends or IBM Quantum |
| 6 | `optimizers/` | classical minimisers: in-repo ADAM and SPSA, scipy wrapper |
| 7 | `vqe.py` | the loop that ties 3-6 together and records every evaluation |
| 8 | `results.py`, `plotting.py` | persistence to `results/` and convergence figures |

Everything is driven by a `RunConfig` (`config.py`) loaded from TOML and exposed through the
`kagome-vqe` command line (`cli.py`).

## Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                   # creates .venv from uv.lock
uv run kagome-vqe exact                   # exact spectrum: -18, -18, -16.9613
uv run kagome-vqe run configs/statevector_hva_adam.toml
uv run kagome-vqe benchmark               # RVB/HVA x ADAM/SPSA on the statevector simulator
```

Point any config at a different backend or tweak any field without editing files:

```bash
uv run kagome-vqe run configs/statevector_hva_adam.toml --backend-override aer_noisy:FakeGuadalupeV2
uv run kagome-vqe run configs/statevector_hva_adam.toml --set optimizer.maxiter=30 --set backend.shots=1024
uv run kagome-vqe run configs/ibm_hva_spsa.toml --dry-run     # ISA depth, gate counts, projected evaluations
uv run kagome-vqe benchmark --backend-override fake:FakeSherbrooke --set backend.shots=2048
uv run kagome-vqe plot results/<dir-a> results/<dir-b> --out convergence.png
uv run kagome-vqe backends --min-qubits 12          # fake backends bundled with qiskit-ibm-runtime
uv run kagome-vqe backends --ibm                    # plus your IBM Quantum devices (needs .env)
```

## Backends

`backend.kind` selects where energies are evaluated:

| kind | what it is | use it for |
|---|---|---|
| `statevector` | exact expectation values of the logical circuit, no shots | reference numbers, fast iteration, tests |
| `aer_ideal` | Aer simulation that samples measurement shots, without noise | checking shot-noise sensitivity of an optimiser |
| `aer_noisy` | as `aer_ideal`, plus the gate, decoherence and readout noise of the fake backend `name` | the "FakeXXX simulator" benchmark; fast enough for full runs |
| `fake` | the fake backend `name` driven through qiskit-ibm-runtime's local testing mode | rehearsing a hardware run with the exact code path (sessions, options) |
| `ibm` | a real IBM Quantum device `name` via the cloud | production runs |

`aer_noisy`, `fake` and `ibm` require `backend.name` (e.g. `FakeGuadalupeV2`, `ibm_torino`).
Every kind except `statevector` requires `backend.shots`.

All shot-based kinds measure the circuit in the Hamiltonian's three measurement bases and
average sampled outcomes, as hardware does, so shot noise and readout error are real. Each
call uses a fresh simulator seed drawn from the run seed, so runs are reproducible while
successive evaluations are independent. After optimisation, the final parameters are
re-measured once with `backend.final_shots`; that re-measured energy is the reported result.
The standard error reported with each energy is the primitive's conservative bound, roughly
ten times the real spread for this Hamiltonian.

Whatever the backend, the ansatz is transpiled with `generate_preset_pass_manager` and the
Hamiltonian is moved onto the chosen physical qubits with `observable.apply_layout`, so the
energy is the same as on the logical circuit up to noise. This is verified in the test suite.

## Optimisers and gradients

`optimizer.type` picks the update rule and `optimizer.gradient` how ADAM and plain gradient
descent obtain their gradient:

| setting | energies per iteration | notes |
|---|---|---|
| `type = "spsa"` | 2 | SPSA's own update rule; `lr` and `perturbation` |
| `type = "adam"`, `gradient = "forward"` | n + 1 | the paper's ADAM; `fd_eps` must be ~0.1 under shot noise |
| `type = "adam"`, `gradient = "central"` | 2n + 1 | less bias than forward differences |
| `type = "adam"`, `gradient = "spsa"` | 2 | SPSA-ADAM hybrid |
| `type = "adam"`, `gradient = "parameter_shift"` | 2K | exact up to shot noise; K = 36 for HVA, 12 for RVB |
| `type = "gd"` | as for its gradient | plain gradient descent, a control for comparing gradients |
| `type = "scipy"` | method dependent | COBYLA, Powell, ...; one energy per call, no batching |

Every run also records the noise-free energy of its parameters after each iteration
(`track_ideal_energy`, on by default up to 20 qubits). On noisy backends this shows how good the
parameters are, separately from the noise floor.

## Error mitigation

`[mitigation]` in a run config switches on any combination of four techniques. They are applied
in this order: counts-level processing, then zero-noise extrapolation, then rescaling.

```toml
[mitigation]
readout = true          # calibrate per-qubit readout errors, correct every shot
symmetry = true         # keep only shots with n/2 ones in each basis (RVB and HVA only)
zne = { scales = [1, 3, 5], extrapolation = "exponential" }   # fold CX gates, fit to zero noise
rescale = { twin = "auto" }   # divide out the damping measured on a classically known twin
```

- **Readout correction** and **symmetry filtering** need individual shots, so they replace the
  Estimator primitive with a sampler that measures each basis itself. Its standard errors are
  exact, not the primitive's bound.
- **Symmetry filtering** relies on the RVB and HVA states being total-spin singlets for every
  parameter value. A singlet has zero magnetisation along every axis, so in every measurement
  basis a noise-free shot has exactly six ones out of twelve.
- **ZNE** multiplies the cost per energy by the number of scale factors.
- **Twin rescaling** measures its twin once per run. The automatic twin needs a classical
  simulation, which is fine up to about 20 qubits.

Every technique works on Aer and through the IBM Quantum sampler. To measure what a stack
reports at a state whose energy is known, evaluate at fixed parameters instead of optimising:

```bash
uv run kagome-vqe benchmark configs/m2_mitigation.toml --point 0 --repeats 8
```

## Results directory

Each run writes `results/<UTC timestamp>-<name>/`:

| file | contents |
|---|---|
| `config.toml` | the resolved `RunConfig`; replay with `kagome-vqe run results/<dir>/config.toml` |
| `result.json` | re-measured final energy and its relative error, exact energy, total shots, parameters, ISA stats, timings |
| `trace.csv` | one row per energy evaluation: `eval_idx, iteration, energy, std, shots, wall_time_s, job_id` |
| `params.npy` | parameters of every evaluation, shape `(nfev, n)` (when `record_params = true`) |
| `ideal_trace.csv` | noise-free energy of the parameters after every iteration, with shots spent |
| `environment.json` | package versions, Python, platform, git commit and dirty flag |
| `run.log` | the log of the run |

`kagome-vqe benchmark` additionally writes `results/<name>-summary/` with `benchmark.json`,
`benchmark.md` and `convergence.png`.

## Benchmark

Every number below is the optimiser's final parameters re-measured once, not the lowest value
seen during optimisation, which is biased low under shot noise.

### Exact statevector

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

### Noisy FakeGuadalupeV2

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

### Gradient comparison (roadmap M1)

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

### Error mitigation (roadmap M2)

Produced by `kagome-vqe benchmark configs/m2_mitigation.toml --point 0 --repeats 8` and
`--point 0.3`, plus `--set backend.noise=readout` and `--set backend.noise=gates` to apply one
error source at a time. Each cell gives the relative bias of the mean against the noise-free
energy, in %, for HVA / RVB, from 8 repeats at 4096 shots per basis on `FakeGuadalupeV2`. The
spread between repeats is 0.03 to 0.15 in energy. ⚠ marks a mean significantly below -18, which
by the variational principle is an artefact.

| stack | full noise, exact params | full noise, θ = 0.3 | readout noise only, θ = 0.3 | gate noise only, θ = 0.3 | shots per energy | one-off shots |
|---|---|---|---|---|---|---|
| none | +28.91 / +29.41 | +28.56 / +28.97 | +8.02 / +8.44 | +22.15 / +22.38 | 12,288 | 0 |
| readout | +22.12 / +22.72 | +22.07 / +22.33 | -0.03 / -0.02 | +22.04 / +22.33 | 12,288 | 32,768 |
| symmetry | +8.67 / +9.03 | +7.89 / +8.54 | +0.69 / +0.69 | +4.46 / +4.99 | 12,288 | 0 |
| readout+symmetry | +5.02 / +5.18 | +4.08 / +4.75 | -0.07 / +0.06 | +4.30 / +4.91 | 12,288 | 32,768 |
| zne | +10.25 / +10.11 | +9.49 / +9.24 | +8.02 / +8.44 | +1.45 / +0.88 | 36,864 | 0 |
| readout+zne | +1.73 / +1.66 | +1.40 / +0.92 | -0.03 / -0.02 | +1.39 / +0.94 | 36,864 | 98,304 |
| readout+symmetry+zne | -7.62⚠ / -7.82⚠ | -8.14 / -8.19⚠ | -0.07 / +0.06 | -7.51 / -7.72⚠ | 36,864 | 98,304 |
| twin | -5.72⚠ / -0.20 | -6.20 / -0.61 | -0.79 / +0.43 | -5.29 / -1.14 | 12,288 | 49,152 |
| readout+twin | -5.45⚠ / -0.81⚠ | -5.56 / -1.19 | -0.66 / -0.07 | -5.14 / -1.01 | 12,288 | 81,920 |
| **readout+symmetry+twin** | +0.27 / +0.36 | -0.34 / -0.15 | -0.06 / -0.06 | +0.27 / +0.43 | 12,288 | 81,920 |

- **Chosen stack: readout correction, then symmetry filtering, then twin rescaling.** Its bias
  is at most 0.43 % in every setting: full noise at both points, readout noise only and gate
  noise only. A method that works on each error source separately is not relying on errors
  cancelling. It costs no extra shots per energy, plus 81,920 one-off shots (readout
  calibration and twin).
- **Why the twin works after filtering.** Filtering leaves only noise inside the singlet sector.
  That residual appears to damp every singlet state by about the same factor, which is exactly
  what rescaling assumes. Without the filter, HVA's twin overshoots by 5 to 6 %.
- **Readout correction plus symmetry filtering alone leaves a 4 to 5 % gate-noise residual,**
  which the filter cannot see. Its standard error comes from a bootstrap over shots and agrees
  with the spread between repeats.
- **Fallback for ansatzes without the singlet symmetry: readout correction plus ZNE,** at 0.9
  to 1.7 % bias in every setting and three times the shots per energy.
- **Never combine symmetry filtering with ZNE.** Post-selection removes noise
  non-multiplicatively, and the extrapolation over-corrects by about 8 % even with gate noise
  alone.
- **Limits.** Correct-then-filter works on the full outcome distribution, which caps it at about
  20 qubits. The automatic twin needs a classical simulation, which has the same limit. Both
  need replacing before larger lattices (M3).

The chosen stack in a full optimisation (`configs/m2_mitigated_vqe.toml`), with the
M1-recommended optimisers, on full `FakeGuadalupeV2` noise:

| run | final energy, re-measured | relative error | noise-free energy of final parameters | total shots |
|---|---|---|---|---|
| HVA, plain SPSA | -17.958 ± 0.116 | 0.23 % | -17.995 | 1,372,160 |
| RVB, ADAM with parameter shift | -17.898 ± 0.068 | 0.57 % | -17.961 | 1,608,704 |

Unmitigated, these runs end near -12.7 (29 % error). Neither reading lies below -18.

**Correction history.** The first M2 result (commit `c9272e3`) recommended readout plus symmetry
alone, applied in the opposite order: filter first, then correct each shot. The documentation
worktree showed that this was two errors cancelling. With readout noise alone, it over-corrected
by about 9 % (-19.57 instead of -18). Under full noise that over-correction happened to cancel
the gate-noise residual on this device. Commit `56aadd3` reversed the order and added the
noise-source switch. The original results remain in `results/m2-evaluation-*` and
`results/m2_vqe-summary`.

## Roadmap

Unmitigated noisy runs plateau near -12.7 because of circuit and readout noise. Milestones M0
(an honest noisy simulator), M1 (better gradients) and M2 (error mitigation, which brings the
bias below 0.5 % on every error source separately) are done. [`docs/roadmap.md`](docs/roadmap.md) records the
diagnosis, the improvement directions (parameter-shift gradients, depth compression, error
mitigation, subspace methods such as QSE and Krylov diagonalisation) and the milestone plan.

## Physics notes

* The open 12-site kagome star has an **exact dimer-product ground state**: six singlets on
  the six spokes give energy -18. There are exactly two perfect matchings of the graph, which
  is why the ground state is two-fold degenerate.
* **Site order matters.** The RVB and HVA circuits act on a ring of qubits (singlets on
  `(2k, 2k+1)`, bond gates on `(i, i+1)`). They reach -18 only when consecutive qubits host
  neighbouring lattice sites, i.e. when the ring follows a Hamiltonian cycle of the kagome
  graph. The 2022 notebook got this implicitly: its hand-written 16-qubit Hamiltonian's ring
  was the Guadalupe heavy-hex loop, and the V1 `Estimator` ignored the transpiler layout. With
  the identity site order the same ansatzes start at -9 instead of -18. This package makes the
  choice explicit through `lattice.site_order`, defaulting to the Hamiltonian cycle
  `[0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]`.
* **eSWAP quirk.** The original notebook's "eSWAP" equals `X ⊗ I · exp(-i φ/2 SWAP)`; it
  carries a stray X gate. The package default is the true `exp(-i φ/2 SWAP)`; the original is
  available with `ansatz.gate = "legacy"`.

The full story, including the layout analysis of the legacy 16-qubit lattice, is in
[`docs/physics.md`](docs/physics.md).

## Running on IBM Quantum

```bash
cp .env.example .env            # fill in IBM_QUANTUM_TOKEN (and IBM_QUANTUM_INSTANCE)
uv run kagome-vqe backends --ibm
uv run kagome-vqe run configs/ibm_hva_spsa.toml --backend-override fake:FakeTorino   # rehearsal, same code path
uv run kagome-vqe run configs/ibm_hva_spsa.toml --set backend.name=ibm_torino         # the real thing
```

Credentials only ever come from the environment or the git-ignored `.env`. The `ibm` and
`fake` kinds share one runtime adapter: execution `mode` (`session`, `batch`, `job`),
`resilience_level` (0-2), `shots`, `max_time`, and a retry loop that resubmits jobs on
transient runtime errors. Because `qiskit_ibm_runtime.EstimatorV2` is deprecated in
qiskit-ibm-runtime 0.50, `backend.estimator_impl` switches between it (`legacy_v2`, default)
and its client-side successor (`executor`). See [`docs/ibm.md`](docs/ibm.md).

## Development

```bash
uv sync --all-groups
uv run pre-commit install
just lint        # ruff check + ruff format --check + mypy
just test        # fast unit tests
just test-all    # plus the slow and fake-backend integration tests
```

Pytest markers: `slow` (full optimisations, tens of seconds), `fake` (noisy fake backends),
`ibm` (real hardware; skipped unless `KAGOME_RUN_IBM=1`). CI runs ruff, mypy and the fast
suite on Python 3.12, 3.13 and 3.14. Code conventions are in
[`docs/STYLE.md`](docs/STYLE.md); how to add ansatzes, optimisers and backends is in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Provenance

* A. W. Shaffar, A. A. Hakim, H. Jung and H. Shin, "Comparative Analysis of Variational
  Ansatzes in the Kagome Heisenberg Antiferromagnet", KICS Fall Conference 2023, pp. 106-107.
* IBM Quantum Open Science Prize 2022:
  <https://github.com/qiskit-community/open-science-prize-2022>.

The original Qiskit 0.42 notebook and helper modules are kept locally under `references/`
(git-ignored). Their API (`qiskit.opflow`, `qiskit.algorithms`, qiskit-nature lattice models,
`qiskit.pulse`) is no longer installable on current Python; this package is a from-scratch
reimplementation on Qiskit 2.x.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
