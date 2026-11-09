"""Configuration and pipeline integration of lattices, graph_hva and mapping strategies."""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from kagome_vqe.ansatz import GraphHVAnsatz, HVAnsatz
from kagome_vqe.config import (
    AnsatzConfig,
    BenchmarkConfig,
    LatticeConfig,
    MappingConfig,
    RunConfig,
    load_run_config,
)
from kagome_vqe.lattice import KAGOME_HAMILTONIAN_CYCLE
from kagome_vqe.vqe import prepare, run_vqe

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = sorted((ROOT / "configs" / "examples").glob("*.toml"))
LEGACY = sorted((ROOT / "configs").glob("*.toml"))


def _load_any(path: Path) -> RunConfig | BenchmarkConfig:
    data = tomllib.loads(path.read_text())
    if "combos" in data or "seeds" in data or path.stem.startswith("benchmark"):
        return BenchmarkConfig.model_validate(data)
    return RunConfig.model_validate(data)


@pytest.mark.parametrize("path", LEGACY, ids=lambda p: p.name)
def test_legacy_configs_still_load(path: Path) -> None:
    cfg = _load_any(path)
    runs = cfg.run_configs() if isinstance(cfg, BenchmarkConfig) else [cfg]
    for run in runs:
        assert run.mapping.strategy == "generic"
        lattice = run.lattice.build()
        assert lattice.site_order == KAGOME_HAMILTONIAN_CYCLE  # unchanged default
        run.ansatz.build(lattice)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_configs_load_and_build(path: Path) -> None:
    cfg = _load_any(path)
    runs = cfg.run_configs() if isinstance(cfg, BenchmarkConfig) else [cfg]
    for run in runs:
        lattice = run.lattice.build()
        ansatz = run.ansatz.build(lattice)
        assert ansatz.num_parameters >= 1
        assert RunConfig.model_validate(run.to_toml_dict()) == run  # TOML round trip


def test_default_lattice_is_unchanged() -> None:
    lattice = LatticeConfig().build()
    assert lattice.name == "kagome_unit_cell"
    assert lattice.site_order == KAGOME_HAMILTONIAN_CYCLE
    assert isinstance(AnsatzConfig().build(lattice), HVAnsatz)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"kind": "kagome_periodic"}, "needs lattice.rows and lattice.cols"),
        ({"kind": "kagome_unit_cell", "rows": 2, "cols": 2}, "do not apply"),
        ({"kind": "custom", "num_sites": 4}, "needs lattice.num_sites and lattice.bonds"),
        ({"kind": "kagome_kvqe20", "bonds": [[0, 1]]}, "only apply to kind='custom'"),
        ({"kind": "kagome_kvqe20", "site_order": "hamiltonian_cycle"}, "only defined"),
        ({"kind": "kagome_bm_rectangle", "rows": 2, "cols": 4, "drop": [[0, 0]]}, "drop"),
    ],
)
def test_lattice_config_validation(data: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        LatticeConfig.model_validate(data)


def test_lattice_config_builds_errors_for_invalid_graphs() -> None:
    with pytest.raises(ValueError, match="not connected"):
        LatticeConfig(kind="custom", num_sites=4, bonds=[[0, 1], [2, 3]]).build()
    with pytest.raises(ValueError, match="not a bond"):
        LatticeConfig(kind="kagome_unit_cell", dimers=[(0, 3), (1, 2), (4, 5)]).build()


def test_lattice_config_dimer_modes() -> None:
    star = LatticeConfig(kind="kagome_unit_cell").build()
    auto = LatticeConfig(kind="kagome_unit_cell", dimers="auto").build()
    assert star.dimers == ((0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11))
    assert auto.dimers is not None
    assert auto.dimers[0][0] == 0


def test_graph_only_fields_rejected_for_other_ansatzes() -> None:
    with pytest.raises(ValidationError, match="only apply to type='graph_hva'"):
        AnsatzConfig(type="hva", include_h0=True)
    with pytest.raises(ValidationError, match="explicit"):
        AnsatzConfig(type="graph_hva", schedule="explicit")


def test_graph_hva_config_builds_expected_counts() -> None:
    lattice = LatticeConfig(kind="kagome_periodic", rows=2, cols=2).build()
    plain = AnsatzConfig(type="graph_hva", reps=3).build(lattice)
    with_h0 = AnsatzConfig(type="graph_hva", reps=3, include_h0=True).build(lattice)
    assert isinstance(plain, GraphHVAnsatz)
    assert plain.num_parameters == 3 * 24
    assert with_h0.num_parameters == 3 * 25
    explicit = AnsatzConfig(
        type="graph_hva",
        schedule="explicit",
        layers=[list(layer) for layer in plain.schedule.layers][::-1],
    ).build(lattice)
    assert isinstance(explicit, GraphHVAnsatz)
    assert explicit.schedule.layers == plain.schedule.layers[::-1]


def test_mapping_config_validation() -> None:
    with pytest.raises(ValidationError, match="geometric strategies"):
        MappingConfig(strategy="generic", physical_layout=[0, 1])
    with pytest.raises(ValidationError, match=r"mapping\.physical_layout"):
        RunConfig.model_validate(
            {
                "mapping": {"strategy": "prb105"},
                "backend": {"kind": "aer_noisy", "name": "FakeNighthawk", "initial_layout": [0]},
            }
        )


def test_symmetry_mitigation_allowed_for_graph_hva() -> None:
    RunConfig.model_validate(
        {
            "ansatz": {"type": "graph_hva"},
            "backend": {"kind": "aer_ideal"},
            "mitigation": {"symmetry": True},
        }
    )


def test_benchmark_combo_mapping_override() -> None:
    bench = BenchmarkConfig.model_validate(
        {
            "lattice": {"kind": "kagome_bm_rectangle", "rows": 2, "cols": 4},
            "combos": [
                {"ansatz": {"type": "graph_hva"}, "mapping": {"strategy": "prb105"}},
                {"ansatz": {"type": "graph_hva"}},
            ],
        }
    )
    runs = bench.run_configs()
    assert [r.mapping.strategy for r in runs] == ["prb105", "generic"]
    assert runs[0].name.endswith("_prb105")


def test_prepare_records_mapping_and_matches_logical_energy() -> None:
    cfg = RunConfig.model_validate(
        {
            "lattice": {"kind": "kagome_bm_rectangle", "rows": 2, "cols": 4},
            "ansatz": {"type": "graph_hva", "reps": 2, "schedule": "prb105"},
            "mapping": {"strategy": "prb105"},
            "backend": {"kind": "statevector"},
        }
    )
    prep = prepare(cfg)
    assert prep.problem.mapping is not None
    assert prep.problem.mapping["status"] == "paper_exact"
    summary = prep.summary()
    assert summary["ansatz"]["nominal_parameters"] == 2 * 10
    from kagome_vqe.estimators import StatevectorEnergy
    from kagome_vqe.transpile import compile_for_backend

    theta = np.random.default_rng(0).uniform(-1, 1, (3, prep.ansatz.num_parameters))
    with (
        StatevectorEnergy(prep.problem) as routed,
        StatevectorEnergy(
            compile_for_backend(prep.ansatz.build(), prep.hamiltonian, None)
        ) as logical,
    ):
        np.testing.assert_allclose(
            routed.evaluate(theta).energies, logical.evaluate(theta).energies, atol=1e-9
        )


def test_prepare_unsupported_mapping_fails_clearly() -> None:
    from kagome_vqe.mapping import MappingError

    cfg = RunConfig.model_validate(
        {
            "lattice": {"kind": "kagome_periodic", "rows": 2, "cols": 2},
            "ansatz": {"type": "graph_hva"},
            "mapping": {"strategy": "prb106_grid"},
        }
    )
    with pytest.raises(MappingError, match="periodic"):
        prepare(cfg)


def test_graph_hva_vqe_reaches_ground_state_of_custom_graph() -> None:
    cfg = load_run_config(ROOT / "configs" / "examples" / "08_custom_graph.toml").model_copy(
        update={"output_dir": Path("unused")}
    )
    result = run_vqe(cfg)
    assert result.final_relative_error < 1e-4
    assert result.ideal_fidelity is not None
    assert result.ideal_fidelity > 0.999


def test_parameter_shift_with_graph_hva_runs() -> None:
    cfg = RunConfig.model_validate(
        {
            "lattice": {"kind": "kagome_bm_rectangle", "rows": 2, "cols": 4},
            "ansatz": {"type": "graph_hva", "reps": 1},
            "optimizer": {"type": "adam", "maxiter": 3, "gradient": "parameter_shift"},
            "backend": {"kind": "statevector"},
            "track_ideal_energy": False,
        }
    )
    result = run_vqe(cfg)
    assert result.optimal_params.size == 10
    assert result.nit == 3
    # each gradient shifts every one of the 30 rotation occurrences (3 per exchange gate) twice
    assert result.nfev >= 3 * 2 * 30


# -- command line ---------------------------------------------------------------------------


def test_cli_lattice_describes_and_plots(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from kagome_vqe.cli import app

    out = tmp_path / "lattice.png"
    result = CliRunner().invoke(
        app, ["lattice", str(ROOT / "configs/examples/08_custom_graph.toml"), "--plot", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "graph_hva" in result.output
    assert out.stat().st_size > 0


def test_cli_exact_with_config() -> None:
    from typer.testing import CliRunner

    from kagome_vqe.cli import app

    result = CliRunner().invoke(
        app, ["exact", "--config", str(ROOT / "configs/examples/02_graph_hva_no_h0.toml")]
    )
    assert result.exit_code == 0, result.output
    assert "-21.7795" in result.output


def test_cli_map_compiles_all_strategies(tmp_path: Path) -> None:
    import json

    from typer.testing import CliRunner

    from kagome_vqe.cli import app

    result = CliRunner().invoke(
        app,
        [
            "map",
            str(ROOT / "configs/examples/03_graph_hva_with_h0.toml"),
            "--set",
            "ansatz.reps=1",
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    rows = json.loads((tmp_path / "mapping.json").read_text())
    status = {row["strategy"]: row["status"] for row in rows}
    assert status == {
        "prb105": "paper_exact",
        "prb106_grid": "paper_inspired",
        "generic": "generic",
    }


def test_cli_map_reports_unsupported(tmp_path: Path) -> None:
    import json

    from typer.testing import CliRunner

    from kagome_vqe.cli import app

    result = CliRunner().invoke(
        app,
        ["map", str(ROOT / "configs/examples/02_graph_hva_no_h0.toml"), "--out", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    rows = {r["strategy"]: r for r in json.loads((tmp_path / "mapping.json").read_text())}
    assert rows["prb105"]["status"] == "unsupported"
    assert "periodic" in rows["prb105"]["reason"]
    assert rows["generic"]["status"] == "generic"
