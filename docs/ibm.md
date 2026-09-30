# Running on IBM Quantum

A checklist for taking a config from the simulator to real hardware.

## 1. Credentials

1. Sign in at <https://quantum.cloud.ibm.com>, open your account page and create an API key.
2. Note the Cloud Resource Name (CRN) of the instance you want to charge (optional if your
   account has a single instance).
3. Copy the template and fill it in; `.env` is git-ignored and never written by the code:

```bash
cp .env.example .env
```

```
IBM_QUANTUM_TOKEN=<api key>
IBM_QUANTUM_INSTANCE=crn:v1:bluemix:public:quantum-computing:us-east:a/...::
IBM_QUANTUM_CHANNEL=ibm_quantum_platform
IBM_QUANTUM_REGION=
```

`kagome_vqe.config.Settings` reads these (prefix `IBM_QUANTUM_`). The old `ibm_quantum`
channel is retired; `ibm_quantum_platform` is the only live one. If no token is set, the code
falls back to credentials previously stored with `QiskitRuntimeService.save_account`.

Verify:

```bash
uv run kagome-vqe backends --ibm --min-qubits 12
```

## 2. Pick a device

Any device with at least 12 qubits works; the transpiler chooses the qubits unless you pass
`backend.initial_layout`. Modern heavy-hex devices (Eagle/Heron: `ibm_sherbrooke`,
`ibm_torino`, ...) all contain 12-qubit loops. Leave `initial_layout` unset on them and let
the layout pass pick a low-error loop; use `--dry-run` to see what it chose.

## 3. Dry run

```bash
uv run kagome-vqe run configs/ibm_hva_spsa.toml --set backend.name=ibm_torino --dry-run
```

This transpiles for the real device (needs credentials, submits nothing) and prints the ISA
statistics: physical qubits, depth, two-qubit gate count, and `projected_evaluations`. It is
the moment to decide whether the depth is acceptable.

## 4. Choose shots and resilience

| `resilience_level` | what the runtime does |
|---|---|
| 0 | nothing |
| 1 | TREX readout-error mitigation and gate twirling (cheap, recommended default) |
| 2 | level 1 plus zero-noise extrapolation (several circuits per evaluation; slower) |

`shots` is per expectation value (`default_shots`). 4096 is a reasonable starting point.
SPSA's `perturbation` must exceed the shot-noise scale of the energy; `0.01` (the paper's
noiseless value) is far too small on hardware, `0.1` is the shipped default for `ibm`
configs.

## 5. Execution mode

| `mode` | behaviour |
|---|---|
| `session` | one reservation for the whole optimisation; iterations are not re-queued. Recommended for VQE. |
| `batch` | jobs are grouped but may interleave with other users' work |
| `job` | every evaluation batch is an independent queued job |

`max_time` bounds a session/batch (`"1h"`, `"90m"`, or seconds). Keep it below your plan's
limit and above `job_timeout_s * expected jobs`.

Every optimiser iteration is submitted as **one** job: the ADAM gradient batch (`n + 1`
parameter vectors) or the SPSA pair is one PUB with broadcast parameters. With HVA + SPSA,
`maxiter = 50` is about 51 jobs.

## 6. Retries

`RuntimeEnergy.evaluate` resubmits on transient failures only: runtime job failures, job
timeouts, `IBMRuntimeError`/`IBMApiError`, and connection errors. Programming errors
(`TypeError`, `ValueError`, ...) propagate immediately. Between attempts it waits
`min(60, 2**attempt)` seconds, up to `backend.retries` times. A job that is still running when
a retry is triggered is cancelled first. If the session reports itself closed, a new one is
opened; that re-enters the queue and is logged as a warning.

`job_timeout_s` is passed to `job.result(timeout=...)` on cloud jobs.

## 7. Estimator implementation

qiskit-ibm-runtime 0.50 deprecates `qiskit_ibm_runtime.EstimatorV2` in favour of the
client-side `qiskit_ibm_runtime.executor_estimator.Estimator`. Both are supported:

```toml
[backend]
estimator_impl = "legacy_v2"   # default today; will be removed upstream
# estimator_impl = "executor"  # the successor; different options model, no seed_estimator
```

Both were verified in local testing mode against `FakeGuadalupeV2`. When upgrading
qiskit-ibm-runtime past 0.50, switch the default and drop `legacy_v2`.

## 8. Rehearse first

`backend.kind = "fake"` runs the identical runtime code path (options, session, retries)
against a fake backend, offline:

```bash
uv run kagome-vqe run configs/ibm_hva_spsa.toml --backend-override fake:FakeTorino --set optimizer.maxiter=2
```

(`resilience_level` has no effect in local mode; the runtime warns about it.)

## 9. Cost estimate

Cost scales with `evaluations x shots`. Per 100 iterations: ADAM needs `100 * (n + 1)`
evaluations, i.e. 1300 for RVB and 300 for HVA; SPSA needs 200 for either. On hardware prefer
HVA (2 parameters, shallower circuit) with SPSA.

## 10. Reading results

`results/<dir>/trace.csv` has one row per evaluation with `job_id`, so any point can be traced
back to the runtime job; `result.json` stores the estimator description (backend, mode, shots,
resilience level) and the ISA statistics; `environment.json` stores the qiskit-ibm-runtime
version.

## 11. Troubleshooting

| symptom | likely cause |
|---|---|
| `IBMNotAuthorizedError` / account errors at startup | wrong or expired `IBM_QUANTUM_TOKEN`, or an instance CRN you cannot access |
| complaints about channel `ibm_quantum` | retired channel; use `ibm_quantum_platform` |
| `unknown fake backend` | fake backends are added and retired with every qiskit-ibm-runtime release; run `kagome-vqe backends --fake` for the current list |
| job never finishes | raise `job_timeout_s` or lower `shots`/`resilience_level`; check `max_time` |
| energies look random on hardware | `perturbation` (SPSA) or `fd_eps` (ADAM) below shot noise; raise them |
