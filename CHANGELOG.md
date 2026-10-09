# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `docs/roadmap.md`: diagnosis of the noisy-energy floor, improvement directions
  (parameter-shift gradients, depth compression, error mitigation, subspace methods) and the
  milestone plan M0 to M5 with benchmark checks.
- README noisy FakeGuadalupeV2 benchmark table.

### Fixed

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
