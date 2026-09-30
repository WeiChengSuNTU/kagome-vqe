# Contributing

## Setup

```bash
uv sync --all-groups
uv run pre-commit install
```

## Checks

```bash
just lint        # ruff check, ruff format --check, mypy
just test        # fast unit tests (excludes slow / ibm markers)
just test-all    # adds the slow statevector runs and the fake-backend smoke tests
```

Or directly: `uv run ruff check .`, `uv run mypy`, `uv run pytest -m "slow or fake"`.

## Adding things

**A new ansatz.** Implement the `Ansatz` protocol (`src/kagome_vqe/ansatz/base.py`): `name`,
`num_qubits`, `num_parameters`, `build()`. Subclass `RingAnsatz` if it starts from the singlet
ring. Export it from `ansatz/__init__.py`, add its name to the `type` literal in
`AnsatzConfig` and its branch in `AnsatzConfig.build()` (`config.py`). Tests go in
`tests/unit/test_ansatz.py`: parameter count, and the energy at zero parameters.

**A new optimiser.** Implement the `Optimizer` protocol (`optimizers/base.py`): `name`,
`maxiter`, `evaluations_per_iteration()`, `minimize()`. Use `as_objective()` and call
`objective.batch()` once per iteration so hardware runs stay one job per iteration. Export
it, extend `OptimizerConfig.type` and `OptimizerConfig.build()`. Tests in
`tests/unit/test_optimizers.py`: convergence on the quadratic and the evaluation count.

**A new backend kind.** Implement `EnergyEstimator` (`estimators.py`): `description`,
`evaluate()`, `__enter__`/`__exit__`. Add the kind to `BackendKind` and to
`make_estimator()`, and teach `backends.resolve_backend()` how to obtain the backend object.
Tests in `tests/unit/test_transpile_estimators.py` (dispatch, batch shape) and, if it needs a
device, a `fake`-marked smoke test in `tests/integration/`.

**A new lattice.** Add a constructor in `lattice.py` returning a `Lattice` with a sensible
default `site_order`, extend `LatticeConfig.kind`, and test bond count, degrees and the exact
spectrum.

## Pull request checklist

- [ ] `just lint` and `just test` pass locally
- [ ] new behaviour has a test; physics numbers have a comment saying where they come from
- [ ] public functions have Google docstrings
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] no credentials, tokens or `results/` artefacts in the diff
