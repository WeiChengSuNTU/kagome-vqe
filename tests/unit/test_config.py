import math
from pathlib import Path

import pytest
import tomli_w
from pydantic import ValidationError

from kagome_vqe.ansatz import HVAnsatz, LibraryAnsatz, RVBAnsatz
from kagome_vqe.config import (
    AnsatzConfig,
    BackendConfig,
    BenchmarkConfig,
    LatticeConfig,
    OptimizerConfig,
    RunConfig,
    Settings,
    apply_overrides,
    load_benchmark_config,
    load_run_config,
    load_toml,
    parse_override,
)
from kagome_vqe.lattice import KAGOME_HAMILTONIAN_CYCLE
from kagome_vqe.optimizers import Adam, ScipyOptimizer, Spsa

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_defaults_reproduce_the_paper() -> None:
    cfg = RunConfig()
    assert cfg.seed == 42
    assert cfg.x0_scale == pytest.approx(math.pi / 4)
    assert cfg.lattice.build().site_order == KAGOME_HAMILTONIAN_CYCLE
    assert cfg.optimizer.maxiter == 100
    assert cfg.optimizer.lr == 0.01
    assert cfg.backend.kind == "statevector"


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.toml")), ids=lambda p: p.name)
def test_shipped_configs_parse(path: Path) -> None:
    if "combos" in load_toml(path):
        bench = load_benchmark_config(path)
        assert len(bench.run_configs()) >= 1
    else:
        load_run_config(path)


def test_lattice_site_orders() -> None:
    assert LatticeConfig(site_order="identity").build().site_order is None
    custom = list(reversed(range(12)))
    assert LatticeConfig(site_order=custom).build().site_order == tuple(custom)


def test_ansatz_factory() -> None:
    lattice = LatticeConfig().build()
    assert isinstance(AnsatzConfig(type="rvb").build(lattice), RVBAnsatz)
    assert isinstance(AnsatzConfig(type="hva").build(lattice), HVAnsatz)
    aware = AnsatzConfig(type="hva", lattice_aware=True).build(lattice)
    assert isinstance(aware, HVAnsatz)
    assert len(aware.extra_bonds) == 6  # the hexagon bonds that are not on the ring
    assert aware.num_parameters == 3
    assert isinstance(AnsatzConfig(type="efficient_su2").build(lattice), LibraryAnsatz)


def test_optimizer_factory(rng) -> None:
    assert isinstance(OptimizerConfig(type="adam").build(rng), Adam)
    assert isinstance(OptimizerConfig(type="spsa").build(rng), Spsa)
    assert isinstance(OptimizerConfig(type="scipy", method="Powell").build(rng), ScipyOptimizer)


def test_backend_requires_name_for_devices() -> None:
    with pytest.raises(ValidationError, match=r"requires backend\.name"):
        BackendConfig(kind="ibm")
    with pytest.raises(ValidationError, match=r"requires backend\.name"):
        BackendConfig(kind="aer_noisy")
    assert BackendConfig(kind="fake", name="FakeGuadalupeV2").label() == "fake:FakeGuadalupeV2"


def test_sampling_backends_require_shots() -> None:
    with pytest.raises(ValidationError, match=r"requires backend\.shots"):
        BackendConfig(kind="aer_ideal", shots=None)
    assert BackendConfig(kind="statevector", shots=None).remeasure_shots is None
    assert BackendConfig(kind="aer_ideal").remeasure_shots == 16384
    assert BackendConfig(kind="aer_ideal", shots=100, final_shots=None).remeasure_shots == 100


def test_validation_ranges() -> None:
    with pytest.raises(ValidationError):
        BackendConfig(resilience_level=3)
    with pytest.raises(ValidationError):
        OptimizerConfig(maxiter=0)
    with pytest.raises(ValidationError):
        RunConfig(name="has space")
    with pytest.raises(ValidationError):
        RunConfig(unknown_field=1)  # type: ignore[call-arg]


def test_overrides() -> None:
    assert parse_override("backend.shots=1024") == (["backend", "shots"], 1024)
    assert parse_override('backend.name="ibm_torino"') == (["backend", "name"], "ibm_torino")
    assert parse_override("backend.name=ibm_torino") == (["backend", "name"], "ibm_torino")
    assert parse_override("lattice.site_order=[1, 0]") == (["lattice", "site_order"], [1, 0])
    with pytest.raises(ValueError, match=r"key\.path=value"):
        parse_override("nonsense")
    data = apply_overrides(
        {"backend": {"kind": "statevector"}}, ["backend.kind=aer_ideal", "seed=7"]
    )
    assert data == {"backend": {"kind": "aer_ideal"}, "seed": 7}


def test_toml_round_trip(tmp_path: Path) -> None:
    cfg = RunConfig(name="rt", backend=BackendConfig(kind="aer_noisy", name="FakeGuadalupeV2"))
    path = tmp_path / "c.toml"
    with path.open("wb") as fh:
        tomli_w.dump(cfg.to_toml_dict(), fh)
    assert load_run_config(path) == cfg


def test_benchmark_expands_to_named_runs() -> None:
    bench = BenchmarkConfig(name="b")
    names = [c.name for c in bench.run_configs()]
    assert names == ["b-rvb_spsa", "b-rvb_adam", "b-hva_spsa", "b-hva_adam"]


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IBM_QUANTUM_TOKEN", "abc")
    monkeypatch.setenv("IBM_QUANTUM_INSTANCE", "crn:x")
    settings = Settings(_env_file=None)
    assert settings.has_credentials
    assert settings.token is not None
    assert settings.token.get_secret_value() == "abc"
    assert "abc" not in repr(settings)
    assert settings.channel == "ibm_quantum_platform"
    monkeypatch.delenv("IBM_QUANTUM_TOKEN")
    assert not Settings(_env_file=None).has_credentials
