"""M2 mitigation on PRB 106 layer noise: readout error, readout correction, symmetry, twin."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from kagome_vqe.config import RunConfig
from kagome_vqe.estimators import LogicalNoiseEnergy
from kagome_vqe.mitigation.counts import ReadoutCalibration, _all_bitstrings, _sector_energy
from kagome_vqe.noise import measurement_bases
from kagome_vqe.vqe import evaluate_fixed, prepare

RING4 = {"kind": "custom", "num_sites": 4, "bonds": [[0, 1], [1, 2], [2, 3], [3, 0]]}
RECT6 = {"kind": "kagome_bm_rectangle", "rows": 2, "cols": 3}


def _cfg(p: float = 0.0, q: float = 0.0, lattice: Any = RECT6, **extra: Any) -> RunConfig:
    return RunConfig.model_validate(
        {
            "name": "m2-test",
            "seed": 5,
            "lattice": lattice,
            "ansatz": {"type": "graph_hva", "reps": 1},
            "optimizer": {"type": "spsa", "maxiter": 1},
            "backend": {
                "kind": "logical_noise",
                "shots": 4096,
                "final_shots": 4096,
                "logical_noise": {
                    "channel": "depolarizing",
                    "probability": p,
                    "readout_error": q,
                    "fast_kernels": True,
                },
            },
            **extra,
        }
    )


def test_readout_error_damps_two_body_terms_and_correction_removes_it() -> None:
    q = 0.04
    raw = evaluate_fixed(_cfg(q=q), 0.3, repeats=6)
    corrected = evaluate_fixed(_cfg(q=q, mitigation={"readout": True}), 0.3, repeats=6)
    damping = (1 - 2 * q) ** 2
    assert raw.mean == pytest.approx(damping * raw.ideal_energy, abs=0.08)
    assert corrected.mean == pytest.approx(corrected.ideal_energy, abs=0.08)
    # the reported standard error is the real one (delta method), not a bound
    assert np.mean(corrected.stds) == pytest.approx(corrected.spread, rel=0.6)


def test_symmetry_filter_and_correct_then_filter_match_the_counts_module() -> None:
    prep = prepare(_cfg(p=0.02, q=0.03, mitigation={"readout": True, "symmetry": True}))
    est = LogicalNoiseEnergy(
        prep.problem,
        shots=2048,
        seed=1,
        readout_error=0.03,
        readout=True,
        symmetry=True,
        fast_kernels=True,
    )
    inverses = est._calibrate()
    rng = np.random.default_rng(2)
    counts = rng.multinomial(2048, rng.dirichlet(np.ones(64)))
    n = 6
    bits = _all_bitstrings(n)
    mask = bits.sum(axis=1) == n // 2
    for _, diagonal in measurement_bases(prep.problem.observable):
        expected, _ = _sector_energy(counts.astype(float), diagonal, mask, inverses)
        mean, var = est._basis_statistics(counts, diagonal, inverses)
        assert mean == pytest.approx(expected, rel=1e-10)
        assert var > 0
    # symmetry alone: the plain filtered mean
    est_sym = LogicalNoiseEnergy(prep.problem, shots=2048, seed=1, symmetry=True)
    _, diagonal = measurement_bases(prep.problem.observable)[0]
    mean, _ = est_sym._basis_statistics(counts, diagonal, None)
    assert mean == pytest.approx(float(counts[mask] @ diagonal[mask]) / counts[mask].sum())
    calibration = ReadoutCalibration(*est._calibration)  # type: ignore[misc]
    assert np.allclose(calibration.inverse_matrices(), inverses)


def test_symmetry_removes_most_gate_noise_bias() -> None:
    raw = evaluate_fixed(_cfg(p=0.01), 0.3, repeats=4)
    sym = evaluate_fixed(_cfg(p=0.01, mitigation={"symmetry": True}), 0.3, repeats=4)
    assert abs(sym.bias) < 0.3 * abs(raw.bias)


def test_twin_rescaling_runs_on_logical_noise() -> None:
    result = evaluate_fixed(_cfg(p=0.01, mitigation={"rescale": {"twin": "auto"}}), 0.3, repeats=2)
    assert "noisy_twin_energy_per_setup" in result.description
    assert abs(result.bias) < 0.5


def test_readout_error_needs_the_trajectory_engine() -> None:
    with pytest.raises(ValueError, match="trajectories"):
        RunConfig.model_validate(
            {
                "backend": {
                    "kind": "logical_noise",
                    "shots": 10,
                    "logical_noise": {"readout_error": 0.02, "engine": "aer"},
                },
                "ansatz": {"type": "graph_hva"},
                "lattice": RING4,
            }
        )
    assert "readout q=0.02" in _cfg(q=0.02).backend.label()


def test_zero_readout_error_draws_no_extra_randomness() -> None:
    prep = prepare(_cfg(p=0.02))
    theta = np.full((1, prep.problem.circuit.num_parameters), 0.3)
    a = LogicalNoiseEnergy(prep.problem, shots=512, seed=3, fast_kernels=True).evaluate(theta)
    b = LogicalNoiseEnergy(
        prep.problem, shots=512, seed=3, fast_kernels=True, readout_error=0.0
    ).evaluate(theta)
    assert a.energies[0] == b.energies[0]


def test_mitigation_study_writes_the_m2_table(tmp_path: Path) -> None:
    from kagome_vqe.mitigation_study import MitigationStudyConfig, run_mitigation_study

    backend = _cfg(p=0.01, q=0.02, lattice=RING4).backend.model_dump(mode="json")
    cfg = MitigationStudyConfig.model_validate(
        {
            "name": "m2small",
            "lattice": RING4,
            "ansatz": {"type": "graph_hva", "reps": 1},
            "points": [{"label": "g", "value": 0.3}],
            "noise": [{"label": "full", "backend": backend}],
            "stacks": [
                {"label": "none"},
                {"label": "readout+symmetry", "mitigation": {"readout": True, "symmetry": True}},
            ],
            "repeats": 3,
        }
    )
    out = run_mitigation_study(cfg, output_root=tmp_path)
    import pandas as pd

    points = pd.read_csv(out / "points.csv")
    assert list(points["stack"]) == ["none", "readout+symmetry"]
    assert abs(points["bias"].iloc[1]) < abs(points["bias"].iloc[0])
    assert points["setup_shots"].iloc[1] == 2 * 16384
    report = (out / "report.md").read_text()
    assert "| readout+symmetry |" in report
