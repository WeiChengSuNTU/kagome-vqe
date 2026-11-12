"""PRB 106 paper-layer noise: channels, insertion, simulators, noisy runs, fixed-point analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from qiskit.quantum_info import DensityMatrix, Kraus, Statevector, SuperOp

from kagome_vqe.ansatz import GraphHVAnsatz
from kagome_vqe.ansatz.graph_hva import NOISE_MARK
from kagome_vqe.config import (
    BackendConfig,
    LatticeConfig,
    LogicalNoiseConfig,
    OptimizerConfig,
    RunConfig,
)
from kagome_vqe.estimators import LogicalNoiseEnergy
from kagome_vqe.hamiltonian import exact_ground_space, heisenberg_hamiltonian
from kagome_vqe.noise import (
    PauliChannel,
    PauliTrajectorySimulator,
    exact_noisy_state,
    insert_layer_noise,
    kraus_operators,
    layer_noise_circuit,
    measurement_bases,
    noise_locations,
    pauli_probabilities,
    survival_probability,
)
from kagome_vqe.noise_analysis import NoiseAnalysisConfig, run_noise_analysis, trajectories_for
from kagome_vqe.optimizers import Spsa
from kagome_vqe.results import load_optimal_params, load_run, run_and_save
from kagome_vqe.vqe import prepare

RING4 = {"kind": "custom", "num_sites": 4, "bonds": [[0, 1], [1, 2], [2, 3], [3, 0]]}
RECT6 = {"kind": "kagome_bm_rectangle", "rows": 2, "cols": 3}


def _config(
    lattice: dict[str, object] = RING4,
    *,
    channel: str = "depolarizing",
    p: float = 0.05,
    reps: int = 1,
    include_h0: bool = False,
    engine: str = "trajectories",
    **extra: object,
) -> RunConfig:
    return RunConfig.model_validate(
        {
            "name": "noise-test",
            "lattice": lattice,
            "ansatz": {"type": "graph_hva", "reps": reps, "include_h0": include_h0},
            "backend": {
                "kind": "logical_noise",
                "shots": 512,
                "final_shots": 1024,
                "logical_noise": {"channel": channel, "probability": p, "engine": engine},
            },
            "optimizer": {"type": "spsa", "maxiter": 3},
            **extra,
        }
    )


def _ansatz(lattice: dict[str, object] = RING4, **kwargs: object) -> GraphHVAnsatz:
    built = LatticeConfig.model_validate(lattice).build()
    return GraphHVAnsatz(built, **kwargs)  # type: ignore[arg-type]


# -- 2. channels ----------------------------------------------------------------------------------


@pytest.mark.parametrize("channel", ["depolarizing", "bit_flip"])
@pytest.mark.parametrize("p", [0.0, 0.01, 0.3])
def test_kraus_operators_are_trace_preserving(channel: str, p: float) -> None:
    ops = kraus_operators(channel, p)  # type: ignore[arg-type]
    total = sum(k.conj().T @ k for k in ops)
    assert np.allclose(total, np.eye(2))


def test_depolarizing_matches_paper_eq7_and_qiskit_parametrisation() -> None:
    """(1-p) rho + p/3 (X rho X + Y rho Y + Z rho Z) == qiskit depolarizing_error(4p/3)."""
    from qiskit_aer.noise import depolarizing_error

    p = 0.12
    ours = SuperOp(Kraus(kraus_operators("depolarizing", p)))
    theirs = SuperOp(depolarizing_error(4 * p / 3, 1))
    assert np.allclose(ours.data, theirs.data)


@pytest.mark.parametrize("channel", ["depolarizing", "bit_flip"])
def test_pauli_channel_instruction_converts_to_the_same_aer_channel(channel: str) -> None:
    p = 0.2
    inst = PauliChannel(*pauli_probabilities(channel, p))  # type: ignore[arg-type]
    assert np.allclose(
        SuperOp(inst.to_aer()).data,
        SuperOp(Kraus(kraus_operators(channel, p))).data,  # type: ignore[arg-type]
    )


def test_bloch_vector_contraction() -> None:
    """Depolarising shrinks every component by 1 - 4p/3; bit flip keeps x, shrinks y and z."""
    p = 0.1
    rho = DensityMatrix(Statevector([np.cos(0.4), np.exp(0.7j) * np.sin(0.4)]))
    paulis = {
        k: np.array(v)
        for k, v in {
            "x": [[0, 1], [1, 0]],
            "y": [[0, -1j], [1j, 0]],
            "z": [[1, 0], [0, -1]],
        }.items()
    }

    def bloch(state: DensityMatrix) -> dict[str, float]:
        return {k: float(np.real(np.trace(m @ state.data))) for k, m in paulis.items()}

    before = bloch(rho)
    dep = bloch(rho.evolve(Kraus(kraus_operators("depolarizing", p))))
    flip = bloch(rho.evolve(Kraus(kraus_operators("bit_flip", p))))
    for k in "xyz":
        assert dep[k] == pytest.approx((1 - 4 * p / 3) * before[k])
    assert flip["x"] == pytest.approx(before["x"])
    assert flip["y"] == pytest.approx((1 - 2 * p) * before["y"])
    assert flip["z"] == pytest.approx((1 - 2 * p) * before["z"])


def test_invalid_probabilities_are_rejected() -> None:
    with pytest.raises(ValueError, match="0, 1"):
        pauli_probabilities("depolarizing", 1.5)
    with pytest.raises(ValueError, match="unknown channel"):
        pauli_probabilities("amplitude_damping", 0.1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="invalid Pauli"):
        PauliChannel(0.6, 0.6, 0.0)


def test_survival_probability_is_eq9_factor() -> None:
    assert survival_probability("bit_flip", 1e-3, 132) == pytest.approx(0.999**132)


# -- 3. insertion locations -----------------------------------------------------------------------


@pytest.mark.parametrize("include_h0", [False, True])
@pytest.mark.parametrize("reps", [1, 2])
def test_noise_after_singlets_and_after_every_layer(reps: int, include_h0: bool) -> None:
    ansatz = _ansatz(RECT6, reps=reps, include_h0=include_h0)
    layers_per_cycle = ansatz.schedule.num_layers + (1 if include_h0 else 0)
    assert ansatz.noise_layers == layers_per_cycle * reps + 1
    circuit = layer_noise_circuit(ansatz, "bit_flip", 0.01)
    assert noise_locations(circuit) == ansatz.num_qubits * ansatz.noise_layers

    # Split the circuit at noise blocks; check what sits between them.
    blocks: list[list[tuple[str, tuple[int, ...]]]] = [[]]
    channel_blocks: list[set[int]] = []
    previous_was_channel = False
    for inst in circuit.data:
        qubits = tuple(circuit.find_bit(q).index for q in inst.qubits)
        if inst.operation.name == "pauli_channel":
            if not previous_was_channel:
                channel_blocks.append(set())
                blocks.append([])
            channel_blocks[-1].add(qubits[0])
            previous_was_channel = True
        else:
            blocks[-1].append((inst.operation.name, qubits))
            previous_was_channel = False
    assert all(b == set(range(ansatz.num_qubits)) for b in channel_blocks)  # every qubit
    assert blocks[-1] == []  # ends with a noise block
    # Block 0: singlet preparation only.
    assert {name for name, _ in blocks[0]} == {"x", "h", "cx"}
    assert {frozenset(q) for name, q in blocks[0] if name == "cx"} == {
        frozenset(d) for d in ansatz.qubit_dimers
    }
    # Every later block: the bonds of exactly one layer (or the H0 block), three rotations each.
    expected: list[set[frozenset[int]]] = []
    previous = None
    for g in ansatz.gates:  # chronological; a new block starts at every (cycle, kind, layer)
        if (g.cycle, g.kind, g.layer) != previous:
            expected.append(set())
            previous = (g.cycle, g.kind, g.layer)
        if not g.pruned:
            expected[-1].add(frozenset(g.qubits))
    assert len(expected) == ansatz.noise_layers - 1
    got = [
        {frozenset(q) for name, q in block if name in ("rxx", "ryy", "rzz")}
        for block in blocks[1:-1]
    ]
    assert got == expected
    for block in blocks[1:-1]:
        assert all(name in ("rxx", "ryy", "rzz") for name, _ in block)


def test_unmarked_build_is_unchanged() -> None:
    ansatz = _ansatz(RECT6, reps=2)
    plain = ansatz.build()
    assert "barrier" not in plain.count_ops()
    marked = ansatz.build(mark_noise_locations=True)
    assert marked.count_ops()["barrier"] == ansatz.noise_layers
    stripped = [i for i in marked.data if i.operation.name != "barrier"]
    assert [i.operation.name for i in stripped] == [i.operation.name for i in plain.data]
    assert all(
        i.operation.label == NOISE_MARK for i in marked.data if i.operation.name == "barrier"
    )


def test_insert_layer_noise_ignores_unlabelled_barriers() -> None:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.barrier()
    qc.barrier(label=NOISE_MARK)
    out = insert_layer_noise(qc, "depolarizing", 0.1)
    assert out.count_ops() == {"barrier": 1, "pauli_channel": 2}


# -- 4. configuration -----------------------------------------------------------------------------


def test_logical_noise_config_round_trips_through_toml() -> None:
    cfg = _config(channel="bit_flip", p=2e-3, engine="aer")
    again = RunConfig.model_validate(cfg.to_toml_dict())
    assert again == cfg
    assert again.backend.logical_noise == LogicalNoiseConfig(
        channel="bit_flip", probability=2e-3, engine="aer"
    )
    assert cfg.backend.label() == "logical_noise[bit_flip p=0.002]"
    assert not cfg.backend.is_exact


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"ansatz": {"type": "hva", "reps": 1}}, "graph_hva"),
        ({"mapping": {"strategy": "prb105"}}, "not routed"),
        ({"mitigation": {"readout": True}}, "mitigation"),
        (
            {"optimizer": {"type": "adam", "gradient": "parameter_shift", "maxiter": 2}},
            "parameter_shift",
        ),
    ],
)
def test_logical_noise_rejects_incompatible_settings(
    update: dict[str, object], message: str
) -> None:
    base = _config().model_dump(mode="json")
    if "ansatz" in update:
        base["lattice"] = {"kind": "kagome_unit_cell"}
    base.update(update)
    with pytest.raises(ValueError, match=message):
        RunConfig.model_validate(base)


def test_backend_logical_noise_field_requires_its_kind() -> None:
    with pytest.raises(ValueError, match="logical_noise"):
        BackendConfig(kind="logical_noise", shots=10)
    with pytest.raises(ValueError, match="logical_noise"):
        BackendConfig(kind="aer_ideal", shots=10, logical_noise=LogicalNoiseConfig())
    with pytest.raises(ValueError, match=r"no backend\.name"):
        BackendConfig(
            kind="logical_noise", name="FakeManilaV2", shots=10, logical_noise=LogicalNoiseConfig()
        )


# -- 5. no hardware assumptions in the logical mode -----------------------------------------------


def test_logical_noise_problem_is_not_routed_or_transpiled() -> None:
    prep = prepare(_config(RECT6, reps=2))
    problem = prep.problem
    assert prep.backend is None
    assert problem.backend_name == "logical"
    assert problem.physical_qubits is None
    assert problem.circuit.num_qubits == prep.lattice.num_sites
    assert set(problem.gate_counts) <= {"x", "h", "cx", "rxx", "ryy", "rzz", "pauli_channel"}
    assert "swap" not in problem.gate_counts
    assert problem.mapping is not None
    assert problem.mapping["status"] == "not_routed"
    assert problem.gate_counts["pauli_channel"] == 6 * prep.ansatz.noise_layers  # type: ignore[attr-defined]


# -- 6. compiled hardware noise acts on the compiled circuit --------------------------------------


def test_hardware_noise_uses_the_compiled_physical_circuit() -> None:
    cfg = RunConfig.model_validate(
        {
            "name": "hw",
            "lattice": RING4,
            "ansatz": {"type": "graph_hva", "reps": 1},
            "backend": {"kind": "aer_noisy", "name": "FakeManilaV2", "shots": 256},
            "optimizer": {"type": "spsa", "maxiter": 1},
        }
    )
    prep = prepare(cfg)
    problem = prep.problem
    assert problem.backend_name == "fake_manila"
    assert prep.backend is not None
    native = set(prep.backend.operation_names) | {"barrier"}
    assert set(problem.gate_counts) <= native  # native gates only, routing included
    assert "rxx" not in problem.gate_counts
    assert problem.circuit.num_qubits == prep.backend.num_qubits
    from kagome_vqe.estimators import make_estimator

    with make_estimator(cfg.backend, problem, prep.backend, seed=1) as estimator:
        assert estimator.description["noise_backend"] == "fake_manila"
        value = estimator.evaluate(np.zeros((1, prep.ansatz.num_parameters))).energies[0]
    assert np.isfinite(value)
    from kagome_vqe.vqe import noise_metadata

    assert noise_metadata(prep)["mode"] == "compiled_hardware"


# -- 1. zero noise and simulator correctness ------------------------------------------------------


def test_zero_noise_reproduces_the_ideal_state_and_energy() -> None:
    ansatz = _ansatz(RECT6, reps=2)
    theta = np.random.default_rng(0).uniform(-1, 1, ansatz.num_parameters)
    circuit = layer_noise_circuit(ansatz, "depolarizing", 0.0)
    sim = PauliTrajectorySimulator(circuit)
    expected = Statevector(ansatz.build().assign_parameters(theta)).data
    assert abs(np.vdot(expected, sim.ideal_state(theta))) == pytest.approx(1.0, abs=1e-12)
    rho = exact_noisy_state(circuit, theta)
    assert np.allclose(rho, np.outer(expected, expected.conj()), atol=1e-10)
    h = heisenberg_hamiltonian(LatticeConfig.model_validate(RECT6).build()).to_matrix(sparse=True)
    sample = sim.sample_expectations(theta, [h], 50, np.random.default_rng(1))
    assert sample.error_free == 50
    assert sample.mean()[0] == pytest.approx(float(np.real(np.vdot(expected, h @ expected))))


def test_zero_noise_estimator_matches_statevector_energy() -> None:
    cfg = _config(RECT6, p=0.0)
    prep = prepare(cfg)
    theta = np.random.default_rng(2).uniform(-1, 1, prep.ansatz.num_parameters)
    state = Statevector(prep.ansatz.build().assign_parameters(theta))
    exact = float(np.real(state.expectation_value(prep.hamiltonian)))
    batch = LogicalNoiseEnergy(prep.problem, shots=20000, seed=3).evaluate(theta[None, :])
    assert abs(batch.energies[0] - exact) < 5 * batch.stds[0]
    assert batch.shots == 20000


@pytest.mark.parametrize("channel", ["depolarizing", "bit_flip"])
def test_trajectories_agree_with_exact_density_matrix(channel: str) -> None:
    ansatz = _ansatz(RECT6, reps=1)
    circuit = layer_noise_circuit(ansatz, channel, 0.04)  # type: ignore[arg-type]
    theta = np.random.default_rng(4).uniform(-1, 1, ansatz.num_parameters)
    hamiltonian = heisenberg_hamiltonian(LatticeConfig.model_validate(RECT6).build())
    h = hamiltonian.to_matrix(sparse=True)
    _, ground = exact_ground_space(hamiltonian)
    rho = exact_noisy_state(circuit, theta)
    energy = float(np.real(np.trace(h @ rho)))
    fidelity = float(np.real(np.trace(ground.conj().T @ rho @ ground)))
    rng = np.random.default_rng(5)
    sample = PauliTrajectorySimulator(circuit).sample_expectations(theta, [h, ground], 40000, rng)
    ci = sample.bootstrap_ci(rng, resamples=400, level=0.999)
    assert ci[0, 0] <= energy <= ci[0, 1]
    assert ci[1, 0] <= fidelity <= ci[1, 1]
    assert 0 < sample.error_free < sample.trajectories


def test_measured_histograms_follow_the_noisy_distribution() -> None:
    ansatz = _ansatz(RING4, reps=1)
    circuit = layer_noise_circuit(ansatz, "depolarizing", 0.08)
    theta = np.array([0.3, -0.7, 1.1, 0.4])[: ansatz.num_parameters]
    rho = exact_noisy_state(circuit, theta)
    shots = 200000
    (zz,) = PauliTrajectorySimulator(circuit).sample_measurements(
        theta, ["ZZZZ"], shots, np.random.default_rng(6)
    )
    expected = np.real(np.diag(rho))
    assert np.max(np.abs(zz / shots - expected)) < 5 * np.sqrt(expected.max() / shots)


def test_measurement_bases_reproduce_the_hamiltonian_expectation() -> None:
    hamiltonian = heisenberg_hamiltonian(LatticeConfig.model_validate(RING4).build())
    state = Statevector(np.random.default_rng(7).normal(size=16) + 0j).data
    state /= np.linalg.norm(state)
    from kagome_vqe.noise import _basis_probabilities

    total = sum(
        float(_basis_probabilities(state[None, :], basis, 4)[0] @ diag)
        for basis, diag in measurement_bases(hamiltonian)
    )
    assert total == pytest.approx(float(np.real(Statevector(state).expectation_value(hamiltonian))))
    assert sorted(b for b, _ in measurement_bases(hamiltonian)) == ["XXXX", "YYYY", "ZZZZ"]


def test_aer_engine_agrees_with_trajectory_engine() -> None:
    prep_t = prepare(_config(RING4, p=0.05))
    prep_a = prepare(_config(RING4, p=0.05, engine="aer"))
    theta = np.full(prep_t.ansatz.num_parameters, 0.4)
    t = LogicalNoiseEnergy(prep_t.problem, shots=40000, seed=1).evaluate(theta[None, :])
    a = LogicalNoiseEnergy(prep_a.problem, shots=40000, seed=1, engine="aer").evaluate(
        theta[None, :]
    )
    assert abs(t.energies[0] - a.energies[0]) < 5 * (t.stds[0] + a.stds[0])
    rho = exact_noisy_state(prep_t.problem.circuit, theta)
    exact = float(np.real(np.trace(prep_t.hamiltonian.to_matrix() @ rho)))
    assert abs(t.energies[0] - exact) < 5 * t.stds[0]


# -- 7./8./9. noisy runs: persistence, budgets, gradient estimators -------------------------------


def test_noisy_spsa_run_saves_noise_metadata_iterates_and_reloads(tmp_path: Path) -> None:
    cfg = _config(
        RING4,
        channel="bit_flip",
        p=0.02,
        optimizer={
            "type": "spsa",
            "maxiter": 50,
            "calibrate": True,
            "calibration_samples": 2,
            "perturbation": 0.1,
            "target_magnitude": 0.2,
        },
        budget={"max_shots": 512 * 3 * 15},
    )
    run_dir = run_and_save(cfg, root=tmp_path)
    run = load_run(run_dir)
    noise = run.metadata["noise"]
    assert noise["mode"] == "paper_layers"
    assert noise["channel"] == "bit_flip"
    built = run.config.ansatz.build(run.config.lattice.build())
    assert isinstance(built, GraphHVAnsatz)
    assert noise["locations"] == 4 * built.noise_layers
    assert "simultaneous" not in run.metadata["optimizer"]["gradient"]
    assert run.metadata["optimizer"]["gradient"].startswith("spsa: built-in two-point")
    assert run.metadata["optimizer"]["hyperparameters"]["calibrate"] is True
    res = run.resources
    assert res["budget_exhausted"] is True
    assert res["optimisation_shots"] <= 512 * 3 * 15
    assert res["objective_evaluations"] == 14  # 4 calibration + 5 x 2; a 6th pair would exceed
    assert res["calibration_evaluations"] == 4
    assert res["circuit_executions"] == (14 + 1) * 3
    assert run.iterates is not None
    assert run.iterates.shape == (res["iterations"] + 1, 4)
    assert np.allclose(run.iterates[0], run.x0)
    assert np.allclose(load_optimal_params(run_dir, num_parameters=4), run.optimal_params)
    assert run.ideal_trace is not None
    assert "step_norm" in run.ideal_trace
    assert run.result["estimator"]["kind"] == "logical_noise"


def test_noisy_adam_uses_the_spsa_gradient(tmp_path: Path) -> None:
    cfg = _config(
        RING4,
        optimizer={
            "type": "adam",
            "gradient": "spsa",
            "lr": 0.02,
            "perturbation": 0.1,
            "maxiter": 3,
        },
    )
    run = load_run(run_and_save(cfg, root=tmp_path))
    assert run.metadata["optimizer"]["gradient"].startswith("spsa: simultaneous perturbation")
    hyper = run.metadata["optimizer"]["hyperparameters"]
    assert hyper["gradient"] == "spsa"
    assert hyper["beta1"] == 0.9
    assert run.resources["objective_evaluations"] == 2 * 3 + 1  # two energies per iteration


# -- SPSA step control ----------------------------------------------------------------------------


def _quadratic(x: np.ndarray) -> float:
    return float(np.sum((x - 3.0) ** 2))


def test_spsa_default_is_unchanged_by_step_control_fields() -> None:
    a = Spsa(maxiter=20, learning_rate=0.1, perturbation=0.1, seed=3).minimize(
        _quadratic, np.zeros(4)
    )
    b = Spsa(
        maxiter=20,
        learning_rate=0.1,
        perturbation=0.1,
        max_step_norm=None,
        clip_update=None,
        seed=3,
    ).minimize(_quadratic, np.zeros(4))
    assert np.array_equal(a.x, b.x)


def test_spsa_max_step_norm_caps_every_update() -> None:
    spsa = Spsa(maxiter=30, learning_rate=1.0, perturbation=0.1, max_step_norm=0.05, seed=1)
    spsa.minimize(_quadratic, np.zeros(4))
    assert max(spsa.step_norms) <= 0.05 + 1e-12
    assert spsa.controlled_steps > 0
    assert max(spsa.raw_step_norms) > 0.05


def test_spsa_clip_update_bounds_each_component() -> None:
    spsa = Spsa(maxiter=10, learning_rate=1.0, perturbation=0.1, clip_update=0.01, seed=1)
    result = spsa.minimize(_quadratic, np.zeros(3))
    assert np.all(np.abs(result.x) <= 10 * 0.01 + 1e-12)
    assert spsa.schedule()["clip_update"] == 0.01


def test_step_control_is_spsa_only_and_described() -> None:
    with pytest.raises(ValueError, match="only apply"):
        OptimizerConfig(type="adam", gradient="spsa", max_step_norm=0.1)
    cfg = OptimizerConfig(type="spsa", calibrate=True, max_step_norm=0.2)
    assert "step norm capped at 0.2" in cfg.gradient_description()
    assert "max_step_norm=0.2" in repr(cfg.build(np.random.default_rng(0)))


# -- fixed-parameter analysis ---------------------------------------------------------------------


def test_trajectory_count_follows_the_paper_rule() -> None:
    cfg = NoiseAnalysisConfig(runs=["x"], expected_error_trajectories=1024, min_trajectories=100)
    p_error = 1 - (1 - 1e-4) ** 132
    assert trajectories_for(cfg, 1e-4, 132) == int(np.ceil(1024 / p_error))
    assert trajectories_for(cfg, 0.0, 132) == 100
    assert trajectories_for(cfg, 0.5, 132) == 1024


def test_fixed_parameter_analysis_reuses_saved_parameters(tmp_path: Path) -> None:
    source = RunConfig.model_validate(
        {
            "name": "noiseless-src",
            "lattice": RING4,
            "ansatz": {"type": "graph_hva", "reps": 1},
            "backend": {"kind": "statevector"},
            "optimizer": {"type": "scipy", "method": "L-BFGS-B", "maxiter": 50},
        }
    )
    run_dir = run_and_save(source, root=tmp_path / "runs")
    cfg = NoiseAnalysisConfig(
        name="fixed-test",
        runs=[str(run_dir)],
        probabilities=[0.0, 0.01, 0.05],
        expected_error_trajectories=200,
        min_trajectories=200,
        shots=256,
        bootstrap=200,
    )
    out = run_noise_analysis(cfg, output_root=tmp_path / "analysis")
    points = json.loads((out / "points.json").read_text())
    assert len(points) == 2 * 3
    theta = load_optimal_params(run_dir)
    for row in points:
        assert row["ideal_energy"] == pytest.approx(load_run(run_dir).result["final_energy"])
        assert row["locations"] == 4 * row["noise_layers"]
        if row["p"] == 0.0:
            assert row["exact_noisy_fidelity"] == pytest.approx(row["ideal_fidelity"])
            assert row["eq9_fidelity"] == pytest.approx(row["ideal_fidelity"])
        else:
            assert row["exact_noisy_fidelity"] < row["ideal_fidelity"]
            assert row["exact_energy_bias"] > 0
    assert theta.size == 4
    assert (out / "report.md").exists()
    assert any((out / "figures").glob("infidelity-*.png"))
    assert (out / "analysis.toml").exists()
    assert json.loads((out / "sources.json").read_text()) == [str(run_dir)]


def test_shipped_noise_analysis_configs_validate() -> None:
    from kagome_vqe.noise_analysis import load_noise_analysis_config

    root = Path(__file__).resolve().parents[2] / "configs" / "noise"
    paths = sorted(root.glob("*.toml"))
    assert paths
    for path in paths:
        cfg = load_noise_analysis_config(path)
        assert cfg.runs
        assert set(cfg.channels) <= {"depolarizing", "bit_flip"}
