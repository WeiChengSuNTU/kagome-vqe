"""End-to-end graph_hva runs: convergence on exact simulators and mapped noisy evaluation."""

from __future__ import annotations

import pytest

from kagome_vqe.config import RunConfig
from kagome_vqe.vqe import evaluate_fixed, run_vqe


def _exact_run(lattice: dict[str, object], reps: int, **ansatz: object) -> RunConfig:
    return RunConfig.model_validate(
        {
            "seed": 1,
            "x0_scale": 0.1,
            "lattice": lattice,
            "ansatz": {"type": "graph_hva", "reps": reps, **ansatz},
            "optimizer": {"type": "scipy", "method": "L-BFGS-B", "maxiter": 400},
            "backend": {"kind": "statevector"},
            "record_params": False,
        }
    )


@pytest.mark.slow
@pytest.mark.parametrize("include_h0", [False, True])
def test_unit_cell_graph_hva_reaches_exact_ground_state(include_h0: bool) -> None:
    result = run_vqe(_exact_run({"kind": "kagome_unit_cell"}, 1, include_h0=include_h0))
    assert result.final_relative_error < 1e-5
    assert result.ideal_fidelity is not None
    assert result.ideal_fidelity > 0.999


@pytest.mark.slow
def test_periodic_torus_improves_with_depth() -> None:
    """12-site torus: deeper circuits get closer; convergence is not required to be exact."""
    errors = [
        run_vqe(
            _exact_run({"kind": "kagome_periodic", "rows": 2, "cols": 2}, p)
        ).final_relative_error
        for p in (1, 3)
    ]
    assert errors[1] < errors[0]
    assert errors[0] < 0.1


@pytest.mark.fake
def test_mapped_noisy_evaluation_with_counts_mitigation() -> None:
    """PRB 105 routing leaves a permutation; readout + symmetry must measure the right qubits."""
    cfg = RunConfig.model_validate(
        {
            "lattice": {"kind": "kagome_bm_rectangle", "rows": 2, "cols": 4},
            "ansatz": {"type": "graph_hva", "reps": 1, "schedule": "prb105_fig3"},
            "mapping": {"strategy": "prb105"},
            "backend": {
                "kind": "aer_noisy",
                "name": "FakeNighthawk",
                "shots": 4000,
                "optimization_level": 1,
            },
            "mitigation": {"readout": True, "symmetry": True, "readout_shots": 4000},
        }
    )
    result = evaluate_fixed(cfg, 0.3, repeats=2)
    kept = result.summary()["kept_fraction"]
    assert kept is not None
    assert 0.5 < kept <= 1.0  # symmetry filtering saw singlet-sector shots
    assert abs(result.bias) < 0.05 * abs(result.ideal_energy)
