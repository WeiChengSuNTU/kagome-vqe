"""End-to-end optimisations. Marked ``slow``/``fake``; run with ``pytest -m "slow or fake"``."""

from pathlib import Path

import pytest

from kagome_vqe.config import AnsatzConfig, BackendConfig, OptimizerConfig, RunConfig
from kagome_vqe.lattice import GUADALUPE_HEAVY_HEX_LOOP
from kagome_vqe.results import new_run_dir, save_run
from kagome_vqe.vqe import run_vqe


@pytest.mark.slow
@pytest.mark.parametrize(
    ("ansatz", "optimizer", "tolerance"),
    [
        ("hva", OptimizerConfig(type="adam", maxiter=60, lr=0.01), 0.01),
        ("hva", OptimizerConfig(type="spsa", maxiter=60, lr=0.01, perturbation=0.01), 0.01),
        ("rvb", OptimizerConfig(type="spsa", maxiter=80, lr=0.01, perturbation=0.01), 0.02),
    ],
)
def test_statevector_reaches_ground_state(
    ansatz: str, optimizer: OptimizerConfig, tolerance: float
) -> None:
    cfg = RunConfig(
        name=f"it_{ansatz}_{optimizer.type}",
        seed=42,
        ansatz=AnsatzConfig(type=ansatz),
        optimizer=optimizer,
        backend=BackendConfig(kind="statevector"),
        record_params=False,
    )
    result = run_vqe(cfg)
    assert result.best_relative_error < tolerance
    assert result.nfev == (
        optimizer.maxiter * (result.optimal_params.size + 1) + 1
        if optimizer.type == "adam"
        else 2 * optimizer.maxiter + 1
    )


@pytest.mark.fake
def test_noisy_fake_guadalupe_smoke(tmp_path: Path) -> None:
    cfg = RunConfig(
        name="fake_smoke",
        seed=3,
        x0_scale=0.1,
        ansatz=AnsatzConfig(type="rvb"),
        optimizer=OptimizerConfig(type="spsa", maxiter=2, lr=0.05, perturbation=0.1),
        backend=BackendConfig(
            kind="aer_noisy",
            name="FakeGuadalupeV2",
            shots=200,
            initial_layout=list(GUADALUPE_HEAVY_HEX_LOOP),
        ),
        output_dir=tmp_path,
    )
    result = run_vqe(cfg)
    assert result.nfev == 5
    assert all(r.shots == 200 for r in result.trace)
    assert all(-18.0 <= r.energy <= 18.0 for r in result.trace)
    assert result.isa_stats["physical_qubits"] == list(GUADALUPE_HEAVY_HEX_LOOP)
    run_dir = save_run(result, new_run_dir(cfg))
    assert (run_dir / "trace.csv").exists()


@pytest.mark.fake
def test_runtime_local_mode_smoke() -> None:
    """Same code path as a real IBM run, on a fake backend; one SPSA iteration."""
    cfg = RunConfig(
        name="runtime_local",
        seed=3,
        x0_scale=0.1,
        optimizer=OptimizerConfig(type="spsa", maxiter=1, lr=0.05, perturbation=0.1),
        backend=BackendConfig(
            kind="fake", name="FakeGuadalupeV2", shots=100, mode="session", resilience_level=1
        ),
        record_params=False,
    )
    result = run_vqe(cfg)
    assert result.nfev == 3
    assert result.estimator["mode"] == "session"
    assert all(r.job_id for r in result.trace)
