# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Experiment persistence and reproducibility (`docs/architecture.md`):
  - `results.run_and_save`: the single run-and-persist entry point (CLI `run`, `benchmark`,
    `replay`, `study`, benchmark scripts). Writes `config.toml`, `environment.json` and
    `status.json` before the run; on failure or Ctrl-C keeps `partial_trace.csv` and records
    the error (`RunFailedError`). `start_run`, `read_status`, `find_runs`,
    `load_optimal_params`; `LoadedRun.status`, `.optimal_params`, `.x0`, `.resources`,
    `.metadata`.
  - `result.json` schema 2: `seed`, `num_parameters`, `resources` (calibration / optimiser /
    re-measurement evaluations, shots, budget, timings with noise-free tracking separated) and
    `metadata` (built lattice with every bond, boundary, dimers; ansatz structure; optimiser
    and the gradient estimator actually used; termination reason). Schema-1 directories load
    unchanged.
  - `[budget] max_evaluations / max_shots` (`BudgetConfig`) on runs and benchmarks, enforced
    on every energy request; a budget stop keeps the last completed iterate.
  - `kagome-vqe replay DIR`, `kagome-vqe params DIR [--out]`.
- Multi-seed experiment matrices (`kagome_vqe.study`, `kagome-vqe study`, `study-summary`):
  lattices x p x H0 x backends x optimisers x seeds from TOML, per-backend budgets,
  `fit_maxiter_to_budget`, resumable studies with a manifest, timed `--dry-run` estimates,
  generated tables, Markdown report and figures. Profiles in `configs/studies/`: optimiser
  smoke, pilot, representative and full profiles, and the graph_hva baseline.
- `OptimizerConfig`: ADAM `beta1`, `beta2`, `amsgrad`; calibrated-SPSA `alpha`, `gamma`,
  `stability_constant`, `calibration_samples`, `target_magnitude`; scipy `options`;
  `gradient_description()`. `Spsa.calibration_evaluations`, `calibrated_learning_rate`.
- Documentation reorganised: README is an overview; `docs/architecture.md`,
  `docs/hardware.md` and `docs/experiments/` (`legacy_m0_m1_m2.md`, moved unchanged from the
  README; `graph_hva_baseline.md`; `graph_hva_optimizers.md`). The raw output of the original
  36-run graph_hva sweep is preserved in `docs/experiments/data/`.

### Changed

- `benchmarks/graph_hva_vqe_sweep.py` runs through the study system and saves every run; its
  JSONL output is unchanged apart from a new `path` key. Its L-BFGS-B `maxfun = 15000`, which
  was scipy's implicit default, is now explicit.

### Fixed

- `ScipyOptimizer` evaluated the objective once more per iteration (for its history) than it
  reported in `nfev`; the callback now receives scipy's intermediate result. This inflated
  the evaluation cost of every scipy method in resource comparisons; optimisation paths are
  unchanged.

- Generalised lattices and the graph-based HVA (`docs/graph_hva.md`):
  - `Lattice` gains optional triangular-lattice `coords`, `boundary`, `wrapped_bonds` and a
    validated `dimers` covering; `find_dimer_covering`, `count_dimer_coverings`.
  - `kagome_lattices`: `kagome_kvqe20`, `kagome_bm_rectangle`, `kagome_patch`,
    `kagome_periodic`, `custom_lattice`; `[lattice] kind` selects them.
  - `schedule`: deterministic edge-colouring (`greedy`), PRB 105 (`prb105`, `prb105_fig3`),
    PRB 106 (`prb106`) and explicit schedules.
  - `GraphHVAnsatz` (`ansatz.type = "graph_hva"`): one parameter per gate, optional per-cycle
    H0 evolution, phase-only and redundant-parameter diagnostics, optional pruning.
  - `mapping` package: `prb105`, `prb106_grid` and `generic` strategies with a shared
    order-preserving SWAP router, 3-CNOT exchange synthesis, SWAP merging, VF2 hardware
    placement with calibration scoring, `paper_exact` / `paper_inspired` / fallback status;
    `[mapping]` config; `transpile.compile_mapping`.
  - `kagome-vqe map` (compile-only strategy comparison), `kagome-vqe lattice`,
    `kagome-vqe exact --config`; `visualize.plot_lattice`; example configs in
    `configs/examples/`.
  - Runs record the noise-free ground-space fidelity (`ideal_fidelity`) and
    `IsaProblem.two_qubit_depth`.

- Roadmap M2, error mitigation (`kagome_vqe.mitigation`, `RunConfig.mitigation`):
  - `CountsEnergy` measures each basis through a sampler, corrects readout errors with
    unbiased per-shot weights, filters shots outside the total-spin-singlet sector, and
    reports exact standard errors.
  - `ZneEnergy` folds two-qubit gates and extrapolates to zero noise with error propagation.
  - `TwinRescaledEnergy` and `choose_twin` divide out the damping measured on a twin with a
    classically known energy.
- `kagome-vqe benchmark --point X --repeats N` evaluates every combo at fixed parameters and
  writes a bias-and-spread table. New config `m2_mitigation.toml`.
- `transpile.fold_two_qubit_gates`; `IsaProblem` keeps the logical observable.
- `backend.noise = "readout" | "gates"` applies one error source at a time on `aer_noisy`.
- Evaluation summaries record the kept fraction of symmetry filtering (`kept_fraction`).
- Configs `m2_mitigation.toml` (fixed-point study) and `m2_mitigated_vqe.toml` (optimisation with
  the chosen readout + symmetry + twin stack).

### Changed

- The runtime session, retry and seeding logic is shared by estimator and sampler
  (`estimators.RuntimeSession`).

- Roadmap M1: gradient estimators separate from update rules (`optimizers.gradients`):
  forward and central differences, simultaneous perturbation, and parameter shift.
  `optimizer.gradient` selects one for ADAM or the new plain gradient descent (`type = "gd"`);
  ADAM with `gradient = "spsa"` is the SPSA-ADAM hybrid.
- Parameter shift per gate occurrence: `ansatz.unshare.unshare_parameters` and `ParameterMap`
  compile the ansatz with one parameter per gate and sum the derivatives per ansatz parameter.
- Noise-free tracking: every run records the statevector energy of its parameters after each
  iteration against shots spent (`ideal_trace.csv`, `ideal_energy` in `result.json`).
- Benchmarks: repeated `seeds`, per-combo `shots`, aggregated tables, and
  `ideal_vs_shots.png`. New configs `m1_gradients_ideal.toml` and `m1_gradients_noisy.toml`.

- `docs/roadmap.md`: diagnosis of the noisy-energy floor, improvement directions
  (parameter-shift gradients, depth compression, error mitigation, subspace methods) and the
  milestone plan M0 to M5 with benchmark checks.
- README noisy FakeGuadalupeV2 benchmark table.

### Fixed

- Readout correction combined with symmetry filtering was biased: filtering first and then
  weighting kept shots over-corrected readout errors by about 9 %, which on FakeGuadalupeV2
  happened to cancel the gate-noise residual (found by the documentation worktree). The
  combination now corrects the outcome distribution first, then filters.

- `docs/physics.md` wrongly stated that parameter-shift gradients do not apply to these
  ansatzes; they apply gate by gate.
- Roadmap M0: `aer_ideal` and `aer_noisy` ignored `shots` and readout error because Aer's
  estimator never sampled measurements, and fixed per-row seeds gave finite differences a
  constant fake slope. Both kinds now sample measurements through `BackendEstimatorV2`, and
  every estimator draws a fresh seed per call from the run seed.
- Shot counts were converted to primitive precision with an off-by-one rounding error.

### Changed

- `run_vqe` re-measures the final parameters once with the new `backend.final_shots`
  (default 16384). `result.json`, benchmark tables and the CLI report that energy and the
  total shot count; the trace minimum is kept only for reference, since it is biased low.
- `backend.shots` is now required for every kind except `statevector`.
- New `backend.aer_method`; the default picks statevector sampling without noise and density
  matrices for noisy circuits up to 12 active qubits.
- `configs/benchmark_fake_guadalupe.toml` uses 4096 shots and a 16384-shot re-measurement.

## [0.1.0] - 2026-09-30

Initial release: a from-scratch Qiskit 2.x reimplementation of the IBM Open Science Prize
2022 kagome VQE notebook and the RVB/HVA ansatz comparison of Shaffar et al. (KICS 2023).

### Added

- `Lattice` value object with explicit `site_order`; `kagome_unit_cell()` defaulting to the
  Hamiltonian-cycle order; legacy `inflated_guadalupe_16()` fixture; `perfect_matchings()`.
- `heisenberg_hamiltonian()` as a `SparsePauliOp`; sparse exact diagonalisation
  (`exact_spectrum`, `exact_ground_energy`).
- Ansatzes: `RVBAnsatz` (true eSWAP, with `gate="legacy"` for the original circuit),
  `HVAnsatz` (optional lattice-aware third layer), `LibraryAnsatz` wrappers.
- Optimisers behind an `Optimizer` protocol: in-repo `Adam` (forward/central differences),
  `Spsa` (fixed gains or Spall calibration), `ScipyOptimizer`; batched evaluation so one
  iteration is one primitive job.
- `compile_for_backend()` with `observable.apply_layout`; backend-agnostic energies.
- Energy estimators for statevector, Aer (ideal / noisy from a fake backend), and
  qiskit-ibm-runtime (fake local mode and real devices) with session/batch/job modes,
  resilience options, retries and the `estimator_impl` switch (`legacy_v2` / `executor`).
- `RunConfig`/`BenchmarkConfig` pydantic models, TOML loading, dotted `--set` overrides,
  `Settings` for IBM credentials from `.env`.
- `kagome-vqe` CLI: `exact`, `run` (with `--dry-run`, `--backend-override`, `--set`),
  `benchmark`, `plot`, `backends`.
- Results directories with `config.toml`, `result.json`, `trace.csv`, `params.npy`,
  `environment.json`, `run.log`; benchmark summaries as JSON and Markdown.
- Shipped configs: statevector HVA+ADAM, noisy FakeGuadalupeV2 RVB+SPSA, IBM HVA+SPSA, and the
  four-combo benchmark.
- Test suite (unit, hypothesis invariants, `slow` and `fake` integration), ruff, mypy strict,
  pre-commit, GitHub Actions on Python 3.12-3.14.
- Documentation: README, `docs/physics.md`, `docs/ibm.md`, `docs/STYLE.md`, CONTRIBUTING,
  CITATION.cff.
