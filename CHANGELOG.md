# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Research overview (branch `research/group-meeting-2026-11`)

- `docs/research_overview.md`: the `graph_hva` studies as one line of argument (geometry → HVA
  → optimiser → noise → mitigation → PIDN → limitations), evidence labels, branch integration,
  provenance of every results directory, comparability rules and the next experiments.
- `benchmarks/geometry_survey.py` and `docs/experiments/data/geometry_survey.csv`: exact
  ground energies, gaps and coordination of the available kagome clusters.
- HVA design study: `configs/studies/graph_hva_design_reference.toml` (noise-free optima,
  `p = 1-4`, H0 off/on, 3 seeds) and `configs/noise/graph_hva_design_fixed_noise.toml` (exact
  layer-noise evaluation), `benchmarks/hva_design_report.py`; data and figure in
  `docs/experiments/`. `p = 2` with H0 off has the lowest noisy fidelity of all `p >= 2`
  ansatzes; H0 at `p = 2` matches `p = 3` in fidelity at `p_e = 1e-3`.
- Removed the never-run full profiles `configs/studies/graph_hva_optimizer_full.toml` and
  `configs/studies/graph_hva_noise_opt_full.toml` (superseded; still in the history of their
  feature branches).
- Added `docs/ibm_runtime_audit.md` from `ibm-hardware-results`.

### Changed

- `graph_hva` with `gradient = "parameter_shift"` now uses the exchange-gate rule: 2 energies per
  exchange gate instead of 6 (one per rotation); gradients are identical, evaluation counts
  are 3x lower. The ring ansatzes are unchanged.
- `zne.scales` must start with 1; `logical_noise` accepts ZNE (probability scaling or layer
  folding) and parameter-shift gradients.

### Fixed

- PIDN Stage III measured its noisy input `C_noisy` unmitigated even when Stage I recorded a
  mitigated cost (readout + symmetry Stage I runs), putting the input ~10 % outside its training
  range; it now uses the Stage I estimator (`pidn.workflow.noisy_input_mitigation`). ZNE and
  relabelled sources were not affected.
- `kagome-vqe study --dry-run` (time estimate) failed for backends with twin rescaling: it now
  passes the twin to the estimator.

### Added

- PIDN niche study on compiled hardware noise (`docs/experiments/graph_hva_pidn_next.md`):
  model gradient descent (`optimizer.type = "mgd"`, `kagome_vqe.optimizers.mgd`, Sung et al.
  2020), heavy-ball momentum for gradient descent (`optimizer.momentum`), a noise-aware PIDN loss
  (`PidnTrainingConfig.noise_aware`: physics targets weighted by `1 / sigma^2` from the
  parameter-shift shot variance, training stopped at the noise floor;
  `pidn.workflow.gradient_variances`, `TrainingBatch.gradient_variance`); benchmarks
  `noise_direction.py` (exact gradient direction and ZNE on FakeNighthawk / FakeKingston /
  FakeGuadalupeV2 compiled noise), `pidn_per_layer_openloop.py`, `pidn_next_optimizers.py`
  (equal-shot optimiser comparison under FakeKingston noise); study configs
  `graph_hva_per_layer_reference.toml`, `graph_hva_per_layer_gd.toml`; tests for the per-layer
  parameter map and the shift rule under sharing with noise.

- PIDN with M2-mitigated Stage I data (`docs/experiments/graph_hva_pidn_m2.md`):
  `kagome_vqe.pidn.relabel` and CLI `pidn-relabel` (re-measure Stage I targets through another
  pipeline on a fixed path), `pidn.diagnostics.target_quality`, `benchmarks/pidn_target_quality.py`,
  `benchmarks/pidn_exact_targets_control.py`; PIDN run names carry the source pipeline; the PIDN
  workflow keeps its estimators open (twin rescaling) and accepts `baseline_studies`; configs
  `configs/pidn/graph_hva_pidn_m2*.toml`, `configs/studies/graph_hva_m2_gd.toml`.
  Follow-up: `benchmarks/pidn_shot_scaling.py` (emulated Stage I at 1x-4096x shots calibrated
  on the real runs, capacity sweep with a memorisation test, Gaussian-smoothed targets,
  closed-loop table; configs `graph_hva_pidn_capacity.toml`, `graph_hva_pidn_shot_scaling.toml`)
  and `benchmarks/pidn_landscape_spectrum.py` (the paper's Gaussian-smoothed contrast and
  low-frequency power ratio on measured landscape slices).
- PIDN (physics-informed denoising network, arXiv:2605.02066) surrogate optimisation
  (`docs/experiments/graph_hva_pidn.md`): `kagome_vqe.pidn` with a numpy autodiff engine
  supporting double backpropagation (`pidn.autodiff`), the dual-branch dual-head network,
  ensembles and the normalised `L_data + beta L_phys` training (`pidn.model`, `pidn.training`),
  non-neural baselines (local quadratic model, last gradient), the three-stage workflow on saved
  ZNE gradient-descent trajectories with cost accounting, trust region and online ZNE anchors
  (`pidn.workflow`), study reports (`pidn.report`) and an open-loop extrapolation test
  (`pidn.diagnostics`, `benchmarks/pidn_extrapolation_check.py`); CLI `pidn`, `pidn-summary`;
  configs in `configs/pidn/`. No new dependency.
- M2-style mitigation on PRB 106 layer noise (`docs/experiments/graph_hva_m2_mitigation.md`):
  optional readout error `logical_noise.readout_error`; readout correction, symmetry (singlet
  sector) filtering and twin rescaling on `logical_noise` (correct-then-filter ratio estimator
  with delta-method standard errors, one-off calibration per estimator);
  `kagome_vqe.mitigation_study` and CLI `mitigation-study` / `mitigation-summary` (fixed-point
  table, fresh setup per repeat, `--workers`); config `configs/mitigation/graph_hva_m2_fixed.toml`.
  Completed like main's M2: `benchmarks/m2_exact_limits.py` (infinite-shot limit of every stack
  from exact density matrices, filter-then-correct, twin damping, predicted spreads),
  `graph_hva_m2_fixed_strong.toml` (`p_e` = 3e-3, 1e-2), full optimisations
  `configs/studies/graph_hva_m2_spsa.toml` and `graph_hva_m2_gd.toml`.
- Error mitigation for `graph_hva` under PRB 106 layer noise
  (`docs/experiments/graph_hva_mitigation.md`):
  - Exchange-gate parameter-shift rule: `unshare_parameters(..., exchange_gates=True)` treats
    `RZZ RYY RXX` with one angle as one occurrence (shift `pi/4`, factor 1; generator gap 2);
    `ParameterMap.shifts` / `.coefficients`; `ParameterShift(include_value=True)`.
  - ZNE on `logical_noise`: `[mitigation.zne] noise_scaling = "probability"` (`p_e -> lambda
    p_e`, `noise.scale_noise_probability`) or `"layer_folding"` (`L -> L (L^dag L)^k` with noise
    after every copy, `noise.fold_noise_layers`); independent seeds per scale; `ZneEnergy`
    takes a custom amplification and returns the per-scale energies (`EnergyBatch.scales`).
  - ZNE gradients by the chain rule over per-scale parameter-shift gradients
    (`mitigation.zne_gradient_weights`, `extrapolation_coefficients`).
  - Gradient trajectories: `optimizer.measure_current`, `RunConfig.track_ideal_gradient`,
    `GradientDescent.records`, `vqe.gradient_trajectory`, `trajectory.npz` per run
    (`LoadedRun.trajectory`); `metadata.final_zne` (per-scale final energies).
  - Fixed-parameter ZNE characterisation `kagome_vqe.zne_analysis`, CLI `zne-analysis`
    (`--dry-run`, `--workers`) and `zne-summary`; configs in `configs/mitigation/`.
  - `kagome_vqe.trajectory_analysis` (gradient cosine / error against the exact noise-free
    gradient, cost curves); study summaries add `trajectories.md` when runs saved trajectories.
  - Studies: `StudyBackend.mitigation`, `StudyConfig.track_ideal_gradient`; budgets and dry-run
    costs count every noise scale; `fit_maxiter_to_budget` for parameter-shift gradients.
  - `logical_noise.fast_kernels` (opt-in): exchange gates as `a I + b SWAP`, butterfly basis
    rotations, cache-sized trajectory batches (~2x faster; reference kernels stay the default
    and reproduce earlier runs bit for bit).
  - Study profile `configs/studies/graph_hva_zne_gd_representative.toml`.

- PRB 106 paper-layer noise (`docs/experiments/graph_hva_noise.md`):
  - `kagome_vqe.noise`: depolarising and bit-flip channels in the paper's parametrisation
    (Eqs. 7-8), `PauliChannel` circuit instruction, `layer_noise_circuit` (channel on every
    qubit after the singlets and after every exchange layer, `n (|c| p + 1)` locations),
    `PauliTrajectorySimulator` (per-shot Pauli errors; only erroneous shots are simulated),
    `exact_noisy_state` (Aer density matrix, up to 12 qubits).
  - Backend kind `logical_noise` with `[backend.logical_noise] channel / probability /
    placement / engine` (`LogicalNoiseConfig`) and the `LogicalNoiseEnergy` estimator
    (trajectory or Aer engine); logical circuit, never routed. `aer_noisy` stays the compiled
    hardware-noise mode.
  - `GraphHVAnsatz.build(mark_noise_locations=True)`, `GraphHVAnsatz.noise_layers`.
  - Fixed-parameter analysis `kagome_vqe.noise_analysis`, CLI `noise-analysis` (with
    `--dry-run`) and `noise-summary`: saved optimal parameters, exact and trajectory noisy
    energy and ground-space fidelity with bootstrap CIs, PRB 106 Eq. 9, optional measured
    energy and optional `aer_noisy` comparison; configs in `configs/noise/`.
- Noisy optimisation (`docs/experiments/graph_hva_noisy_optimizers.md`): study profiles
  `graph_hva_noise_opt_{smoke,pilot,representative,full}.toml` and the noise-free p = 3
  reference `graph_hva_noise_reference.toml`.
- Optional SPSA step control, off by default: `optimizer.max_step_norm`, `clip_update`;
  `Spsa.step_norms`, `raw_step_norms`, `controlled_steps`, `schedule()`.
- Run results record `iterates.npy` (parameters after every iteration), `step_norm` in
  `ideal_trace.csv`, `metadata.noise` (mode, channel, placement, locations),
  `metadata.optimizer.hyperparameters` and `.steps`, `resources.circuit_executions`. Study
  summaries add measured versus noise-free error, peak (excursion) error and step sizes.
- `kagome-vqe study --no-summary` for several processes sharing one study directory.

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

- Documentation: the PRB 105 2x6 rectangle has 16 bonds, not 20 (README, baseline and
  optimiser write-ups, full optimiser profile description). No result changes.
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
