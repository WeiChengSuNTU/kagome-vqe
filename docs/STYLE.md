# Code style

The rules the linters enforce, the ones they cannot, and why.

## Language and layout

* Python 3.12 or newer (`requires-python = ">=3.12"`; `.python-version` pins 3.13 for
  development, CI also runs 3.14).
* `src/` layout: the package is `src/kagome_vqe/`, tests in `tests/`, run configs in
  `configs/`, prose in `docs/`. Nothing is importable without installing the package, which is
  what `uv sync` does.
* One module per pipeline stage (see the table in `README.md`). A module's docstring says what
  *role* it plays in the pipeline, not just what it contains.

## Ruff

`ruff check` and `ruff format` with line length 100, target `py312`. Selected rule families:

| family | purpose |
|---|---|
| `E`, `F`, `W` | pycodestyle / pyflakes basics |
| `I` | import sorting (first-party = `kagome_vqe`) |
| `N` | PEP 8 naming |
| `UP` | modern syntax (`X | None`, `list[int]`, ...) |
| `B` | bugbear: likely bugs |
| `C4`, `SIM`, `PERF` | comprehensions, simplifications, performance |
| `PT` | pytest style |
| `RUF` | ruff-specific checks |
| `D` | pydocstyle, Google convention |
| `ANN` | every function is annotated |
| `NPY` | numpy pitfalls (legacy random API in particular) |
| `PL` | pylint subset |
| `T20` | no `print()` outside the CLI |
| `TID` | tidy imports |

Deliberate ignores (see `pyproject.toml`):

* `D203` / `D213`: mutually exclusive with `D211` / `D212`; we use "no blank line before class
  docstring" and "summary on the first line".
* `PLR0913` (too many arguments): scientific configuration functions legitimately have many
  keyword arguments; keep them keyword-only.
* `PLR2004` (magic values): physics constants and small integers are fine with a comment.
* `ANN401` (`Any`): allowed for `**kwargs` pass-throughs and untyped Qiskit objects.
* `N806`: upper-case locals like `H` for a Hamiltonian are idiomatic.

Tests relax `D`, `ANN`, `PLR` and `T20`. The CLI module is exempt from `T20`.

## Docstrings

Google style on every public module, class, function and method: one-line summary, blank line,
detail, then `Args:`, `Returns:`, `Raises:` as applicable. Explain *why* and the physics where
relevant; the signature already says *what*. Private helpers get a one-liner.

## Typing

* Full type hints; `mypy --strict` on `src/` and `tests/` (tests may omit return types).
* The Qiskit stack (`qiskit`, `qiskit_aer`, `qiskit_ibm_runtime`, `rustworkx`) is largely
  untyped; it is covered by `ignore_missing_imports` overrides. Wrap what crosses the boundary
  in our own typed objects (`IsaProblem`, `EnergyBatch`) rather than letting `Any` spread.
* `__all__` in every module; `py.typed` ships with the package.

## Data and components

* Value objects are `@dataclass(frozen=True, slots=True)` (`Lattice`, `IsaProblem`,
  `TraceRow`, `EnergyBatch`, `IterationInfo`).
* Configuration is pydantic v2, frozen, `extra="forbid"`, with `build()` factories that map
  names to classes in exactly one place (`config.py`).
* Pluggable components are `typing.Protocol`s: `Ansatz`, `Optimizer`, `Objective`,
  `EnergyEstimator`. Implement the protocol; do not subclass a base unless you want its
  behaviour (`RingAnsatz`).
* Pure functions in `lattice.py` and `hamiltonian.py`. Side effects (files, network, console)
  live only in `results.py`, `estimators.py` (backend calls) and `cli.py`.

## Logging and output

* Library code uses `log = logging.getLogger(__name__)` and never prints.
* Only `cli.py` writes to the console (Rich); `logging.py` installs the handlers.

## Randomness and reproducibility

* `numpy.random.Generator` only; never the global `np.random.*` functions.
* One `seed` per run fans out explicitly to the initial point, SPSA directions,
  `seed_transpiler` and simulator seeds. Functions that need randomness take a `Generator` or
  seed argument; they do not create hidden global state.
* Every result directory carries `config.toml` and `environment.json`; `uv.lock` is committed.

## Secrets

* Never in code, configs, tests or logs. Credentials come from `Settings` (`.env`, env vars),
  stored as `pydantic.SecretStr`. `.env` is git-ignored; the reference notebook that leaked
  tokens is git-ignored too.

## Naming

* `snake_case` functions and variables, `PascalCase` classes, `UPPER_CASE` constants.
* Ansatz classes end in `Ansatz` (`RVBAnsatz`, `HVAnsatz`, `LibraryAnsatz`); config models end
  in `Config`; energy estimators end in `Energy` (`StatevectorEnergy`, `AerEnergy`,
  `RuntimeEnergy`); optimisers are plain nouns (`Adam`, `Spsa`, `ScipyOptimizer`).
* Config `type`/`kind` literals are lower-case snake (`"aer_noisy"`, `"hva"`).

## Tests

* pytest; markers `slow`, `fake`, `ibm` declared in `pyproject.toml` (`--strict-markers`).
  Default run excludes `slow` and `ibm`; `ibm` is skipped unless `KAGOME_RUN_IBM=1`.
* One behaviour per test, named for the behaviour (`test_zero_parameters_are_the_exact_ground_state`).
* `pytest.mark.parametrize` over cases; shared objects as fixtures in `tests/conftest.py`.
* Hypothesis for invariants (spectrum invariant under site permutation, eSWAP unitarity).
* Physics assertions carry the expected number and a comment saying where it comes from.

## Git

* Commit subject: imperative mood, at most 72 characters; body explains *why*.
* Branch per change, pull request into `main`; CI must be green.
* Update `CHANGELOG.md` under `[Unreleased]` in the same PR.
* Version lives in `pyproject.toml`; bump on release and move the changelog section.
