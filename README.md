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
| `aer_ideal` | Aer shot-based simulation without noise | checking shot-noise sensitivity of an optimiser |
| `aer_noisy` | Aer with the noise model and coupling map of the fake backend `name` | the "FakeXXX simulator" benchmark; fast enough for full runs |
| `fake` | the fake backend `name` driven through qiskit-ibm-runtime's local testing mode | rehearsing a hardware run with the exact code path (sessions, options) |
| `ibm` | a real IBM Quantum device `name` via the cloud | production runs |

`aer_noisy`, `fake` and `ibm` require `backend.name` (e.g. `FakeGuadalupeV2`, `ibm_torino`).

Whatever the backend, the ansatz is transpiled with `generate_preset_pass_manager` and the
Hamiltonian is moved onto the chosen physical qubits with `observable.apply_layout`, so the
energy is the same as on the logical circuit up to noise. This is verified in the test suite.

## Results directory

Each run writes `results/<UTC timestamp>-<name>/`:

| file | contents |
|---|---|
| `config.toml` | the resolved `RunConfig`; replay with `kagome-vqe run results/<dir>/config.toml` |
| `result.json` | final/best energy, exact energy, relative errors, parameters, ISA stats, timings |
| `trace.csv` | one row per energy evaluation: `eval_idx, iteration, energy, std, shots, wall_time_s, job_id` |
| `params.npy` | parameters of every evaluation, shape `(nfev, n)` (when `record_params = true`) |
| `environment.json` | package versions, Python, platform, git commit and dirty flag |
| `run.log` | the log of the run |

`kagome-vqe benchmark` additionally writes `results/<name>-summary/` with `benchmark.json`,
`benchmark.md` and `convergence.png`.

## Benchmark

### Exact statevector

Seed 42, the paper's hyperparameters (`maxiter = 100`, `lr = 0.01`, `perturbation = 0.01`),
initial point `x0 = pi/4 * rand`. Produced by `kagome-vqe benchmark`.

| run | ansatz | optimizer | final E | best E | exact E | rel. err (best) | evaluations | time (s) |
|---|---|---|---|---|---|---|---|---|
| rvb_spsa | rvb | spsa | -17.8369 | -17.8386 | -18.0000 | 0.897 % | 201 | 5.0 |
| rvb_adam | rvb | adam | -17.9980 | -17.9980 | -18.0000 | 0.011 % | 1301 | 24.7 |
| hva_spsa | hva | spsa | -18.0000 | -18.0000 | -18.0000 | 0.000 % | 201 | 6.4 |
| hva_adam | hva | adam | -18.0000 | -18.0000 | -18.0000 | 0.000 % | 301 | 9.0 |

Evaluation counts follow the optimisers' cost model: ADAM with forward differences spends
`n + 1` evaluations per iteration (13 for RVB, 3 for HVA) plus one final evaluation; SPSA
spends 2 per iteration plus one final evaluation.

### Noisy FakeGuadalupeV2

The table below is produced by `configs/benchmark_fake_guadalupe.toml` (`aer_noisy` on
`FakeGuadalupeV2`, heavy-hex loop layout).

<!-- FAKE_GUADALUPE_TABLE -->

Without error mitigation the noisy energies sit far above -18: the transpiled circuits carry
66 (HVA) to 93 (RVB) CX gates on Guadalupe-era error rates. That gap, not the noiseless
convergence, is the real benchmark for ansatz depth and mitigation settings.

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
