# Architecture

How the package is organised, how one run flows through it, and how results are stored.
Physics conventions are in [`physics.md`](physics.md); lattices, the graph HVA and mapping in
[`graph_hva.md`](graph_hva.md); code conventions in [`STYLE.md`](STYLE.md).

## 1. Pipeline

One run is a pipeline whose stages map one-to-one onto modules in `src/kagome_vqe/`:

| stage | module | responsibility |
|---|---|---|
| 1 | `lattice.py`, `kagome_lattices.py` | the spin graph (sites, bonds, couplings, optional coordinates, boundary, dimer covering) and its generators |
| 2 | `hamiltonian.py` | the graph as a `SparsePauliOp`; exact diagonalisation for reference energies and ground-space fidelity |
| 3 | `schedule.py`, `ansatz/` | gate schedules; trial circuits `RVBAnsatz`, `HVAnsatz`, `GraphHVAnsatz`, library wrappers |
| 4 | `mapping/`, `transpile.py` | embed the logical circuit in hardware connectivity (PRB 105, PRB 106 or Qiskit routing); compile circuit and observable for a backend |
| 5 | `estimators.py`, `mitigation/` | energies on statevector, Aer, fake backends or IBM Quantum; readout, symmetry, ZNE, twin rescaling |
| 6 | `optimizers/` | ADAM, SPSA, gradient descent, scipy; pluggable gradient estimators |
| 7 | `vqe.py` | the loop: initial point, budget, trace of every evaluation, noise-free tracking, resource accounting |
| 8 | `results.py`, `plotting.py` | persistence of runs, summaries, figures |
| 9 | `study.py` | multi-seed experiment matrices on top of 7 and 8 |

Everything is configured by pydantic models in `config.py` (`RunConfig`, `BenchmarkConfig`;
`study.py` adds `StudyConfig`), loaded from TOML and exposed through the `kagome-vqe` command
line in `cli.py`. Benchmark scripts in `benchmarks/` only select configurations; they contain
no execution logic of their own.

## 2. One run

`vqe.prepare(cfg)` builds everything that needs no circuit execution: lattice, Hamiltonian,
ansatz, compiled circuit and observable, exact spectrum, optimiser. `kagome-vqe run --dry-run`
prints exactly this.

`vqe.run_vqe(cfg)` then:

1. draws the initial point `x0 = x0_scale * U[0, 1)^n` from `cfg.seed`
   (`vqe.initial_point`). It depends only on the seed, the scale and the parameter count, so
   every optimiser run with the same seed on the same ansatz starts from the same point;
2. wraps the estimator in an `EnergyObjective` that records every evaluation (`trace`) and
   enforces `cfg.budget` (`max_evaluations`, `max_shots`): a batch that would exceed a limit is
   refused before it is evaluated and the run ends with the last completed iterate;
3. after every optimiser iteration records the noise-free energy of the iterate
   (`ideal_trace`), timed separately so it does not inflate the optimiser's wall time;
4. re-measures the final parameters once with `backend.final_shots`: this is the reported
   energy (the trace minimum is biased low under shot noise);
5. computes the noise-free energy and ground-space fidelity of the final parameters, and the
   resource accounting and metadata described below.

## 3. Results directories

`results.run_and_save(cfg)` is the single entry point that runs and persists; `kagome-vqe run`,
`benchmark`, `replay`, `study` and the scripts in `benchmarks/` all use it. Each run gets a new
directory `results/<UTC timestamp>-<name>/` (a counter is appended on a clash; nothing is
ever overwritten):

| file | written | contents |
|---|---|---|
| `config.toml` | at start | the resolved `RunConfig`; replay with `kagome-vqe replay <dir>` or `kagome-vqe run <dir>/config.toml` |
| `environment.json` | at start | package versions, Python, platform, git commit and dirty flag, command line |
| `status.json` | at start, updated at the end | `running`, then `completed`, `failed` or `interrupted` (with error type, message, traceback and number of completed evaluations) |
| `run.log` | during | the run's log |
| `result.json` | on success | see below |
| `trace.csv` | on success | one row per energy evaluation: `eval_idx, iteration, energy, std, shots, wall_time_s, job_id` |
| `params.npy` | on success, if `record_params` | parameters of every evaluation, shape `(nfev, n)` |
| `ideal_trace.csv` | on success, if tracked | noise-free energy of the iterate after every iteration, with evaluations and shots spent so far |
| `partial_trace.csv` | on failure | the evaluations completed before the failure |

`result.json` (schema version 2) holds:

* energies: `final_energy` (re-measured), `final_std`, `exact_energy`, relative errors,
  `best_energy` (trace minimum, biased), `ideal_energy` (noise-free energy of the final
  parameters) and `ideal_fidelity`;
* parameters: `x0` and `optimal_params`, `num_parameters`, `seed`;
* `nfev`, `nit`, `total_shots`, `wall_time_s`, `optimizer_message`;
* `resources`: `objective_evaluations` split into `calibration_evaluations` (SPSA) and
  `optimizer_evaluations`, `remeasurement_evaluations` (always 1), `iterations`,
  `measurement_groups`, `shots_per_evaluation`, `optimisation_shots`, `total_shots` (all
  measurement bases, mitigation overheads and the re-measurement), the `budget` and whether it
  stopped the run, and timings (`optimiser_wall_time_s` excludes the noise-free tracking);
* `metadata`: the built lattice (every bond with its coupling, boundary, wrapped bonds,
  dimer covering, coordinates, site order), the Hamiltonian convention and exact spectrum,
  the ansatz (type, reps, H0, schedule, parameter count, phase-only and redundant gates), the
  optimiser (label, implementation, **the gradient estimator actually used**), and the
  termination reason (`optimizer` or `budget`);
* ISA statistics and the estimator description.

Older (schema 1) directories have no `status.json`, `resources` or `metadata`; every reader
treats those as optional, and `load_run` loads them unchanged.

Python API (`kagome_vqe.results`):

```python
from kagome_vqe.results import find_runs, load_optimal_params, load_run, run_and_save

run_dir = run_and_save(cfg)  # run + persist (raises RunFailedError on failure)
run = load_run(run_dir)  # config, result, trace, params, ideal_trace
theta = load_optimal_params(run_dir, num_parameters=48)  # checked against the target ansatz
for d in find_runs("results"):
    ...  # completed runs below a directory
```

Reusing optimised parameters without re-optimising, e.g. for a noise or mitigation study:

```python
from kagome_vqe.vqe import evaluate_fixed

cfg = load_run(run_dir).config.model_copy(update={"backend": noisy_backend_cfg})
evaluate_fixed(cfg, load_optimal_params(run_dir), repeats=8)
```

or from the shell: `kagome-vqe params <run dir> --out theta.npy`.

`kagome-vqe benchmark` additionally writes `results/<name>-summary/` (`benchmark.json`,
`benchmark.md`, figures). That summary directory is derived data and is rewritten when a
benchmark of the same name is run again (the milestone documentation looks it up by name);
the run directories it summarises are never overwritten.

## 4. Studies

`study.py` turns a `StudyConfig` (TOML in `configs/studies/`) into a matrix of ordinary run
configurations, `lattices x reps x include_h0 x backends x optimisers x seeds`, and runs each
through `run_and_save`:

```text
results/<UTC timestamp>-<study name>/
    study.toml          resolved study configuration
    environment.json    provenance at study start
    manifest.jsonl      one line per attempted run: labels, status, directory
    runs/<UTC timestamp>-<run name>/   ordinary results directories (section 3)
    summary/            runs.csv/json, groups.csv/json, report.md, figures/
```

* Run names are `<study>-<lattice>-p<p>-<h0|noh0>-<backend>-<optimizer>-s<seed>`; stripping
  `-s<seed>` gives the group that is aggregated over seeds.
* Budgets belong to backends and are identical for every optimiser on that backend;
  `fit_maxiter_to_budget` sets `maxiter` of fixed-cost optimisers (SPSA, ADAM with the SPSA
  gradient) so that calibration, iterations and the final evaluation exactly use it.
* Runs are ordered lattice, p, H0, backend, seed, optimiser: an interrupted study holds
  complete optimiser comparisons for the seeds it finished. `--resume <study dir>` skips
  completed runs and retries failed ones in new directories; `--max-runs` and `--time-limit`
  bound one invocation; `--dry-run` prints the plan with a timed cost estimate.
* `summary/` is regenerated from the saved runs by `kagome-vqe study-summary <study dir>`;
  nothing in it is entered by hand.

## 5. Command line

| command | purpose |
|---|---|
| `kagome-vqe exact [--config C]` | exact spectrum of the configured lattice |
| `kagome-vqe lattice C [--plot F]` | describe or draw a lattice, dimers and schedule |
| `kagome-vqe run C [--dry-run] [--set k=v] [-b kind:name]` | one run, saved under `results/` |
| `kagome-vqe replay DIR` | re-run a saved run's `config.toml` into a new directory and compare |
| `kagome-vqe params DIR [--out F]` | print or export a run's optimised parameters |
| `kagome-vqe benchmark C [--point x]` | ansatz x optimiser combos on one backend; or fixed-point evaluation |
| `kagome-vqe study C [--dry-run] [--resume DIR] [--only S] [--max-runs N] [--time-limit MIN] [--output-dir D]` | multi-seed experiment matrix |
| `kagome-vqe study-summary DIR` | regenerate a study's tables, report and figures |
| `kagome-vqe map C [-b ...]` | compile-only comparison of mapping strategies |
| `kagome-vqe plot DIR... [--kind ideal]` | convergence or noise-free-error-versus-shots figures |
| `kagome-vqe backends [--ibm]` | list fake (and IBM) backends |
