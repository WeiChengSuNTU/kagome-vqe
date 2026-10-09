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

## Results directory

Each run writes `results/<UTC timestamp>-<name>/`:

| file | contents |
|---|---|
| `config.toml` | the resolved `RunConfig`; replay with `kagome-vqe run results/<dir>/config.toml` |
| `result.json` | re-measured final energy and its relative error, exact energy, total shots, parameters, ISA stats, timings |
| `trace.csv` | one row per energy evaluation: `eval_idx, iteration, energy, std, shots, wall_time_s, job_id` |
| `params.npy` | parameters of every evaluation, shape `(nfev, n)` (when `record_params = true`) |
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

## Roadmap

Noisy runs plateau near -12.7 because of circuit and readout noise, not optimiser failure.
Milestone M0, an honest noisy simulator, is done. [`docs/roadmap.md`](docs/roadmap.md) records the
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
