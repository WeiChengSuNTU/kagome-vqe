# kagome-vqe

Variational quantum eigensolver (VQE) for the spin-1/2 Heisenberg antiferromagnet on the
kagome lattice, built on Qiskit 2.x and ready for IBM Quantum hardware. The original target is
the 12-site kagome unit cell; configurable lattices (open patches, periodic tori, custom
graphs), a graph-based Hamiltonian variational ansatz with one parameter per gate, and two
paper-based hardware mappings for square-grid devices extend it
(see [Generalised lattices and hardware mapping](#generalised-lattices-graph-hva-and-hardware-mapping)).

The package rebuilds the IBM Quantum Open Science Prize 2022 kagome challenge, and the
resonating-valence-bond (RVB) and Hamiltonian-variational (HVA) ansatz comparison from
Shaffar et al. (KICS 2023), as a typed, tested, reproducible Python project. One TOML file
describes a run; one flag moves it from an exact simulator to a noisy fake device to a real
IBM Quantum system.

## Supported models

The spin-1/2 Heisenberg antiferromagnet `H = sum over bonds of J (XX + YY + ZZ)` (Pauli
normalisation, `= 4 sum S.S`) on:

| lattice (`[lattice] kind`) | description |
|---|---|
| `kagome_unit_cell` (default) | the open 12-site kagome star (18 bonds; exact ground energy -18, a dimer product) |
| `kagome_bm_rectangle` | `rows x cols` open rectangles of Bosse & Montanaro (PRB 105), e.g. 2x6: 12 sites, 20 bonds |
| `kagome_kvqe20` | the 20-site open patch of Kattemolle & van Wezel (PRB 106) |
| `kagome_patch` | the kagome sites of a `rows x cols` window of triangular-lattice coordinates, optionally with sites dropped |
| `kagome_periodic` | `rows x cols` tori (periodic boundaries), e.g. 2x2 cells: 12 sites, 24 bonds |
| `custom` | any graph: `num_sites` and `bonds = [[a, b], [a, b, J], ...]`, optional coordinates and dimers |

Lattices are validated: no duplicate or self bonds, connected, and a dimer covering must be a
perfect matching of bonds; the site count never implies a geometry. Exact diagonalisation
provides reference energies and ground-space fidelities up to about 20
sites (dense up to 10 qubits, sparse with a residual check above).

## Main features

* **Ansatzes**: the legacy ring ansatzes of the 2022 challenge (`rvb`, `hva`: two shared
  parameters), the graph-based HVA with one parameter per gate (`graph_hva`,
  `N_theta = p * N_b`, optional H0 block), and Qiskit library circuits.
* **Backends**: exact statevector, Aer with shot sampling, Aer with a fake device's noise
  model, fake devices through the runtime code path, and real IBM Quantum devices.
* **Optimisers**: in-repo ADAM, SPSA (plain or calibrated), gradient descent and any scipy
  method; gradients by forward/central differences, simultaneous perturbation or parameter
  shift.
* **Error mitigation**: readout correction, singlet-symmetry filtering, zero-noise
  extrapolation and twin rescaling, in any combination.
* **Hardware mapping**: paper-exact square-grid embeddings (PRB 105, PRB 106) and generic
  routing, with compile-only resource comparisons for IBM Nighthawk and Heron.
* **Reproducibility**: every run is a self-contained results directory (configuration,
  provenance, full trace, initial and final parameters, resource accounting, failure status)
  that can be reloaded, replayed and reused; multi-seed experiment matrices (`kagome-vqe study`)
  with budgets, resumption and generated reports.

## Installation and quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                   # creates .venv from uv.lock
uv run kagome-vqe exact                   # exact spectrum: -18, -18, -16.9613
uv run kagome-vqe run configs/statevector_hva_adam.toml
uv run kagome-vqe benchmark               # RVB/HVA x ADAM/SPSA on the statevector simulator
uv run kagome-vqe study configs/studies/graph_hva_optimizer_smoke.toml   # ~1 min optimiser smoke study
```

Point any config at a different backend or tweak any field without editing files:

```bash
uv run kagome-vqe run configs/statevector_hva_adam.toml --backend-override aer_noisy:FakeGuadalupeV2
uv run kagome-vqe run configs/statevector_hva_adam.toml --set optimizer.maxiter=30 --set backend.shots=1024
uv run kagome-vqe run configs/ibm_hva_spsa.toml --dry-run     # ISA depth, gate counts, projected evaluations
uv run kagome-vqe replay results/<dir>                        # re-run a saved experiment into a new directory
uv run kagome-vqe params results/<dir> --out theta.npy        # reuse optimised parameters elsewhere
uv run kagome-vqe plot results/<dir-a> results/<dir-b> --out convergence.png
uv run kagome-vqe benchmark --backend-override fake:FakeSherbrooke --set backend.shots=2048
uv run kagome-vqe backends --min-qubits 12          # fake backends bundled with qiskit-ibm-runtime
uv run kagome-vqe backends --ibm                    # plus your IBM Quantum devices (needs .env)
```

## Architecture

One TOML file (`RunConfig`) describes a run. The pipeline lattice -> Hamiltonian -> ansatz ->
mapping/transpilation -> estimator -> optimiser -> VQE loop -> results maps one-to-one onto
modules in `src/kagome_vqe/`; `study.py` adds multi-seed experiment matrices on top. Module
responsibilities, the run lifecycle, the results-directory format and the command reference
are in [`docs/architecture.md`](docs/architecture.md).

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

## Ansatzes

| `[ansatz] type` | parameters | notes |
|---|---|---|
| `hva` | 2 (shared per layer) | legacy ring HVA of the 2022 challenge; needs a Hamiltonian-cycle site order |
| `rvb` | 12 | legacy ring resonating-valence-bond ansatz |
| `graph_hva` | `p * N_b` (OPG), `p * (N_b + 1)` with `include_h0` | singlets on a dimer covering, then `p` cycles of exchange gates `exp(-i theta/2 (XX+YY+ZZ))` on the actual bonds; schedules `greedy`, `prb105`, `prb105_fig3`, `prb106`, `explicit` |
| `real_amplitudes`, `efficient_su2` | library | hardware-efficient baselines |

## Generalised lattices, graph HVA and hardware mapping

Everything here is additive: the ring ansatzes and all existing configs behave as before, and
`[mapping] strategy = "generic"` (the default) is the plain transpiler path. Full description,
conventions, paper cross-checks and limitations: [`docs/graph_hva.md`](docs/graph_hva.md).

**Mapping** (`[mapping] strategy`):

| strategy | construction | auxiliary qubits | supports |
|---|---|---|---|
| `prb105` | Bosse & Montanaro square-grid embedding; one SWAP layer before and one after the `H4`/`H5` bonds, 7 layers per cycle | none | open patches with coordinates; `paper_exact` on the PRB 105 rectangles |
| `prb106_grid` | Kattemolle & van Wezel KVQE_G; hexagon centres act as swapping stations, 6 layers per cycle | one per three sites (in `|0>`) | open patches with coordinates; `paper_exact` on the 20-site patch |
| `generic` | 3-CNOT exchange synthesis + Qiskit layout and SABRE routing | none | any graph, any device |

A strategy never substitutes generic routing silently: with `fallback = "error"` (default)
an inapplicable strategy fails with the reason. Inapplicable cases are periodic lattices, no
coordinates, or a device where the grid does not fit, such as heavy-hex Heron. With
`fallback = "generic"` the result is labelled `generic_fallback`. The routed grid is
placed on the device by subgraph matching against the real coupling map, preferring
low-error couplers, and `physical_layout` can fix the placement. Routed circuits are
verified against the logical circuit in noiseless simulation, including the final qubit
permutation and auxiliary qubits returning to `|0>`.

A strategy never substitutes generic routing silently: with `fallback = "error"` (default) an
inapplicable strategy fails with the reason; with `fallback = "generic"` the result is
labelled `generic_fallback`. Example configs (`configs/examples/`):

```bash
uv run kagome-vqe run configs/examples/01_legacy_hva_unit_cell.toml          # 1. original 12-site ring HVA
uv run kagome-vqe run configs/examples/02_graph_hva_no_h0.toml               # 2. graph_hva, no H0 (12-site torus)
uv run kagome-vqe run configs/examples/03_graph_hva_with_h0.toml             # 3. graph_hva with H0 (PRB 105 2x6)
uv run kagome-vqe run configs/examples/04_mapping_prb105.toml --dry-run      # 4. PRB 105 mapping on FakeNighthawk
uv run kagome-vqe run configs/examples/05_mapping_prb106.toml --dry-run      # 5. PRB 106 mapping on FakeNighthawk
uv run kagome-vqe run configs/examples/06_mapping_generic.toml --dry-run     # 6. generic routing
uv run kagome-vqe map configs/examples/07_nighthawk_compile_only.toml        # 7. Nighthawk compile-only comparison
uv run kagome-vqe map configs/examples/07_nighthawk_compile_only.toml -b ibm:ibm_phoenix   # real Target, no job
uv run kagome-vqe lattice configs/examples/08_custom_graph.toml --plot lattice.png        # describe / draw a lattice
uv run kagome-vqe exact --config configs/examples/02_graph_hva_no_h0.toml    # exact spectrum of any configured lattice
uv run kagome-vqe benchmark configs/examples/09_benchmark_graph_hva.toml     # graph_hva vs ring HVA
```

`kagome-vqe map` and `run --dry-run` only compile; with `-b ibm:...` they fetch the device
`Target` but never submit a job.

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

Runs can be capped by `[budget] max_evaluations` / `max_shots`, enforced on every energy
request (calibration included). The M1 recommendations (plain SPSA for the two-parameter
HVA, ADAM with parameter shift for RVB) were established for the legacy ring ansatzes and do
not automatically apply to `graph_hva`; see the optimiser study below.

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

## Experiments

Detailed experiment write-ups (goals, configuration, budgets, seeds, results, reproduction
commands, interpretation, limitations):

| document | contents | status |
|---|---|---|
| [`docs/experiments/legacy_m0_m1_m2.md`](docs/experiments/legacy_m0_m1_m2.md) | 12-site star, legacy ring ansatzes: exact and noisy FakeGuadalupeV2 benchmarks (M0), gradient comparison (M1), error mitigation (M2) | completed |
| [`docs/experiments/graph_hva_baseline.md`](docs/experiments/graph_hva_baseline.md) | `graph_hva` with L-BFGS-B on the star, the PRB 105 2x6 rectangle and the 12-site torus, `p = 1-4`, with and without H0 | completed (2 seeds) |
| [`docs/experiments/graph_hva_optimizers.md`](docs/experiments/graph_hva_optimizers.md) | L-BFGS-B vs calibrated SPSA vs ADAM + SPSA gradient for `graph_hva`, noiseless and finite-shot, equal budgets | smoke and limited comparison completed; full profile configured, not yet run |
| [`docs/hardware.md`](docs/hardware.md) | compile-only mapping benchmarks on IBM Nighthawk and Heron | completed |
| [`docs/roadmap.md`](docs/roadmap.md) | diagnosis of the noisy-energy gap and the milestone plan | living |

Summary of the legacy results: unmitigated noisy runs on FakeGuadalupeV2 plateau near -12.7
(29 % error) because of gate and readout noise; the M2 stack readout correction + symmetry
filtering + twin rescaling brings the bias to +0.4 to +1.9 % without relying on errors
cancelling.

## Current limitations

* Exact references (energies, fidelities, noise-free tracking) and the M2 mitigation stack
  (full-distribution readout correction, automatic twin) need classical simulation and cap at
  about 20 qubits.
* The paper mappings support open patches with coordinates only; heavy-hex devices are
  reported unsupported for them.
* `graph_hva` on the periodic 12-site torus does not reach the ground state at `p <= 4` within
  the evaluation limits used so far.
* Optimiser conclusions for `graph_hva` so far rest on a limited 12-site comparison; the full
  multi-seed study is configured but not run. Physical noise has not yet been studied for
  `graph_hva`.

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
[`docs/physics.md`](docs/physics.md). Lattices beyond the 12-site cell, the graph-based HVA and the hardware mappings are
covered in [`docs/graph_hva.md`](docs/graph_hva.md).

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
just lint        # ruff check + ruff format --check
just typecheck   # mypy
just test        # fast unit tests
just test-all    # plus the slow and fake-backend integration tests
just docs-test   # numbers quoted by report/ and slides/ (needs the local M0-M2 results/)
```

Pytest markers: `slow` (full optimisations, tens of seconds), `fake` (noisy fake backends),
`ibm` (real hardware; skipped unless `KAGOME_RUN_IBM=1`). CI runs ruff, mypy and the fast
suite on Python 3.12, 3.13 and 3.14. Code conventions are in
[`docs/STYLE.md`](docs/STYLE.md); how to add ansatzes, optimisers and backends is in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Provenance

* A. W. Shaffar, A. A. Hakim, H. Jung and H. Shin, "Comparative Analysis of Variational
  Ansatzes in the Kagome Heisenberg Antiferromagnet", KICS Fall Conference 2023, pp. 106-107.
* J. L. Bosse and A. Montanaro, "Probing ground-state properties of the kagome
  antiferromagnetic Heisenberg model using the variational quantum eigensolver",
  Phys. Rev. B 105, 094409 (2022): `prb105` mapping and schedule, per-edge ansatz with H0.
* J. Kattemolle and J. van Wezel, "Variational quantum eigensolver for the Heisenberg
  antiferromagnet on the kagome lattice", Phys. Rev. B 106, 214429 (2022): `prb106_grid`
  mapping, KVQE_G schedule and 20-site patch, one parameter per gate.
* IBM Quantum Open Science Prize 2022:
  <https://github.com/qiskit-community/open-science-prize-2022>.

The original Qiskit 0.42 notebook and helper modules are kept locally under `references/`
(git-ignored). Their API (`qiskit.opflow`, `qiskit.algorithms`, qiskit-nature lattice models,
`qiskit.pulse`) is no longer installable on current Python; this package is a from-scratch
reimplementation on Qiskit 2.x.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
