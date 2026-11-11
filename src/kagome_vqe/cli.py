"""The ``kagome-vqe`` command line.

Commands::

    kagome-vqe exact [--config CONFIG]    exact spectrum of the kagome Hamiltonian
    kagome-vqe lattice CONFIG [--plot P]  describe (and draw) a configured lattice
    kagome-vqe map CONFIG                 compile-only comparison of hardware mappings
    kagome-vqe run CONFIG [--dry-run]     one VQE run from a TOML config
    kagome-vqe benchmark [CONFIG]         RVB/HVA x ADAM/SPSA on one backend
    kagome-vqe plot RESULTS_DIR...        convergence figure from saved runs
    kagome-vqe backends                   list fake (and, with credentials, IBM) backends

Every command accepts ``--set key.path=value`` overrides and ``--backend-override KIND[:NAME]``
so a config written for the statevector simulator can be pointed at a fake backend or a real
device without editing files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from kagome_vqe import __version__
from kagome_vqe.config import (
    BenchmarkConfig,
    RunConfig,
    Settings,
    apply_overrides,
    load_toml,
)
from kagome_vqe.logging import configure_logging

if TYPE_CHECKING:
    from kagome_vqe.vqe import VQERunResult

app = typer.Typer(
    help="VQE for the kagome Heisenberg antiferromagnet.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()
log = logging.getLogger(__name__)

SetOption = Annotated[
    list[str] | None,
    typer.Option("--set", "-s", help="Override a config value, e.g. --set backend.shots=1024."),
]
BackendOverride = Annotated[
    str | None,
    typer.Option(
        "--backend-override",
        "-b",
        help="Replace the backend: KIND[:NAME], e.g. aer_noisy:FakeGuadalupeV2 or ibm:ibm_torino.",
    ),
]
Verbose = Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")]


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"kagome-vqe {__version__}")
        raise typer.Exit


@app.callback()
def _main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version_callback, is_eager=True)
    ] = False,
) -> None:
    """VQE for the kagome Heisenberg antiferromagnet."""


def _backend_overrides(spec: str | None) -> list[str]:
    if spec is None:
        return []
    kind, _, name = spec.partition(":")
    overrides = [f"backend.kind={kind}"]
    if name:
        overrides.append(f'backend.name="{name}"')
    return overrides


def _load(path: Path, overrides: list[str] | None, backend: str | None) -> dict[str, Any]:
    return apply_overrides(load_toml(path), [*(overrides or []), *_backend_overrides(backend)])


def _print_json(data: Any) -> None:
    console.print_json(json.dumps(data, default=str))


# -- exact ------------------------------------------------------------------------------------


@app.command()
def exact(
    k: Annotated[int, typer.Option(help="How many eigenvalues.")] = 3,
    t: Annotated[float, typer.Option(help="Exchange coupling (unit cell only).")] = 1.0,
    config: Annotated[
        Path | None,
        typer.Option(help="Run config TOML whose [lattice] to diagonalise.", dir_okay=False),
    ] = None,
    set_: SetOption = None,
) -> None:
    """Exact diagonalisation of the 12-site kagome cell, or of a configured lattice."""
    import numpy as np

    from kagome_vqe.hamiltonian import exact_spectrum, heisenberg_hamiltonian
    from kagome_vqe.lattice import count_dimer_coverings, kagome_unit_cell

    if config is None:
        lattice = kagome_unit_cell(t)
    else:
        lattice = RunConfig.model_validate(_load(config, set_, None)).lattice.build()
    hamiltonian = heisenberg_hamiltonian(lattice)
    spectrum = exact_spectrum(hamiltonian, k=k)
    degeneracy = int(np.sum(np.isclose(spectrum, spectrum[0], atol=1e-8)))
    table = Table(title=f"{lattice.name}, H = sum_<ij> J_ij (XX + YY + ZZ)")
    table.add_column("quantity")
    table.add_column("value")
    table.add_row("sites / bonds", f"{lattice.num_sites} / {lattice.num_bonds}")
    table.add_row("boundary", lattice.boundary)
    table.add_row("Pauli terms", str(len(hamiltonian)))
    table.add_row("lowest eigenvalues", ", ".join(f"{v:.8f}" for v in spectrum))
    table.add_row("E0 per site (S.S units)", f"{spectrum[0] / 4 / lattice.num_sites:.6f}")
    table.add_row("ground-state degeneracy", str(degeneracy))
    if lattice.num_sites % 2 == 0:
        coverings = count_dimer_coverings(lattice)
        table.add_row("dimer coverings", str(coverings))
        if coverings:
            table.add_row("dimer-product energy", f"{-3 * t * lattice.num_sites / 2:.4f}")
    console.print(table)


@app.command()
def lattice(
    config: Annotated[Path, typer.Argument(help="Run config TOML.", exists=True, dir_okay=False)],
    set_: SetOption = None,
    plot: Annotated[
        Path | None, typer.Option(help="Write a figure of the lattice and schedule here.")
    ] = None,
) -> None:
    """Describe the configured lattice, dimer covering and graph_hva schedule."""
    from kagome_vqe.ansatz import GraphHVAnsatz

    cfg = RunConfig.model_validate(_load(config, set_, None))
    lat = cfg.lattice.build()
    ansatz = cfg.ansatz.build(lat)
    info: dict[str, Any] = {
        "name": lat.name,
        "num_sites": lat.num_sites,
        "num_bonds": lat.num_bonds,
        "boundary": lat.boundary,
        "wrapped_bonds": [list(b) for b in lat.wrapped_bonds],
        "connected": lat.is_connected(),
        "coords": None if lat.coords is None else [list(c) for c in lat.coords],
        "kagome_geometry": lat.is_kagome_geometry(),
        "site_order": lat.site_order,
        "bonds": [[a, b, w] for a, b, w in lat.edges],
        "dimers": None if lat.dimers is None else [list(d) for d in lat.dimers],
    }
    if isinstance(ansatz, GraphHVAnsatz):
        info["graph_hva"] = ansatz.describe()
        info["schedule_layers"] = ansatz.schedule.to_list()
    _print_json(info)
    if plot is not None:
        from kagome_vqe.visualize import plot_lattice

        plot_lattice(
            lat.with_dimers(ansatz.dimers) if isinstance(ansatz, GraphHVAnsatz) else lat,
            schedule=ansatz.schedule if isinstance(ansatz, GraphHVAnsatz) else None,
            out=plot,
        )
        console.print(f"wrote {plot}")


@app.command(name="map")
def map_(
    config: Annotated[Path, typer.Argument(help="Run config TOML.", exists=True, dir_okay=False)],
    strategy: Annotated[
        list[str] | None,
        typer.Option(
            "--strategy",
            help="Strategy to compile (repeatable): prb105, prb106_grid, generic. Default: all.",
        ),
    ] = None,
    set_: SetOption = None,
    backend_override: BackendOverride = None,
    out: Annotated[Path | None, typer.Option(help="Output directory.")] = None,
    verbose: Verbose = False,
) -> None:
    """Compile the configured circuit with each mapping strategy; no job is submitted.

    For backend kind 'ibm' only the device's Target is fetched (credentials needed); nothing
    runs on the device.
    """
    from kagome_vqe.benchmarking import mapping_benchmark, write_mapping_benchmark
    from kagome_vqe.mapping import MAPPING_STRATEGIES

    configure_logging(logging.DEBUG if verbose else logging.WARNING)
    strategies = strategy or list(MAPPING_STRATEGIES)
    unknown = [s for s in strategies if s not in MAPPING_STRATEGIES]
    if unknown:
        msg = f"unknown strategies {unknown}; choose from {MAPPING_STRATEGIES}"
        raise typer.BadParameter(msg)
    cfg = RunConfig.model_validate(_load(config, set_, backend_override))
    rows = mapping_benchmark(cfg, strategies, settings=Settings())
    out_dir = out or Path(cfg.output_dir) / f"{cfg.name}-mapping"
    json_path, md_path = write_mapping_benchmark(rows, out_dir)
    console.print(md_path.read_text())
    console.print(f"summary: {json_path}")


# -- run --------------------------------------------------------------------------------------


@app.command()
def run(
    config: Annotated[Path, typer.Argument(help="Run config TOML.", exists=True, dir_okay=False)],
    set_: SetOption = None,
    backend_override: BackendOverride = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Prepare and report only.")] = False,
    no_progress: Annotated[bool, typer.Option("--no-progress", help="Hide the bar.")] = False,
    verbose: Verbose = False,
) -> None:
    """Run one VQE described by CONFIG and save it under results/."""
    configure_logging(logging.DEBUG if verbose else logging.INFO)
    cfg = RunConfig.model_validate(_load(config, set_, backend_override))
    _run_one(cfg, dry_run=dry_run, progress=not no_progress)


def _run_one(cfg: RunConfig, *, dry_run: bool, progress: bool) -> Path | None:
    from kagome_vqe.results import RunFailedError, run_and_save
    from kagome_vqe.vqe import TraceRow, prepare

    settings = Settings()
    if cfg.backend.kind == "ibm" and not settings.has_credentials:
        log.warning("IBM_QUANTUM_TOKEN not set; relying on credentials saved by qiskit-ibm-runtime")

    prepared = prepare(cfg, settings=settings)
    if dry_run:
        _print_json(prepared.summary())
        return None

    total = prepared.projected_evaluations
    if cfg.budget.max_evaluations is not None:
        total = min(total or cfg.budget.max_evaluations, cfg.budget.max_evaluations)
    with Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}" if total else "{task.completed}"),
        TextColumn("E={task.fields[energy]:.5f}"),
        TimeElapsedColumn(),
        console=console,
        disable=not progress,
    ) as bar:
        task = bar.add_task(cfg.name, total=total, energy=float("nan"))

        def on_eval(row: TraceRow) -> None:
            bar.update(task, advance=1, energy=row.energy)

        captured: list[VQERunResult] = []
        try:
            run_dir = run_and_save(
                cfg,
                prepared=prepared,
                settings=settings,
                on_evaluation=on_eval,
                on_result=lambda result, _: captured.append(result),
            )
        except RunFailedError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
    _print_result(captured[0], run_dir)
    return run_dir


def _print_result(result: VQERunResult, run_dir: Path) -> None:
    cfg = result.config
    table = Table(title=f"{cfg.name} on {cfg.backend.label()}")
    table.add_column("quantity")
    table.add_column("value")
    shots_note = "exact" if result.final_shots is None else f"{result.final_shots} shots"
    table.add_row("exact E0", f"{result.exact_energy:.6f}")
    table.add_row(
        f"final energy (re-measured, {shots_note})",
        f"{result.final_energy:.6f} (std bound {result.final_std:.3f})",
    )
    table.add_row("relative error", f"{100 * result.final_relative_error:.4f} %")
    if result.ideal_energy is not None and not cfg.backend.is_exact:
        table.add_row("noise-free energy of final params", f"{result.ideal_energy:.6f}")
    if result.ideal_fidelity is not None:
        table.add_row("ground-space fidelity", f"{result.ideal_fidelity:.4f}")
    table.add_row("trace minimum (biased low)", f"{result.best_energy:.6f}")
    table.add_row("evaluations / iterations", f"{result.nfev} / {result.nit}")
    table.add_row("total shots", f"{result.total_shots:,}")
    table.add_row("termination", result.optimizer_message)
    table.add_row("wall time", f"{result.wall_time_s:.1f} s")
    table.add_row("saved to", str(run_dir))
    console.print(table)


@app.command()
def replay(
    run_dir: Annotated[
        Path, typer.Argument(help="A results directory.", exists=True, file_okay=False)
    ],
    set_: SetOption = None,
    no_progress: Annotated[bool, typer.Option("--no-progress", help="Hide the bar.")] = False,
    verbose: Verbose = False,
) -> None:
    """Re-run a saved experiment from its config.toml into a new results directory.

    The original directory is never modified. With no --set overrides, a statevector run
    reproduces the original final energy exactly; shot-based runs reproduce it for the same
    package versions (see the saved environment.json).
    """
    from kagome_vqe.results import load_run

    configure_logging(logging.DEBUG if verbose else logging.INFO)
    cfg = RunConfig.model_validate(_load(run_dir / "config.toml", set_, None))
    new_dir = _run_one(cfg, dry_run=False, progress=not no_progress)
    if new_dir is None or not (run_dir / "result.json").exists():
        return
    before, after = load_run(run_dir), load_run(new_dir)
    delta = after.result["final_energy"] - before.result["final_energy"]
    console.print(
        f"final energy: original {before.result['final_energy']:.8f}, replay "
        f"{after.result['final_energy']:.8f} (difference {delta:+.2e})"
    )


@app.command()
def params(
    run_dir: Annotated[
        Path, typer.Argument(help="A results directory.", exists=True, file_okay=False)
    ],
    out: Annotated[
        Path | None, typer.Option(help="Write the vector to this .npy (or .txt) file.")
    ] = None,
) -> None:
    """Print (or export) the optimised parameters of a saved run, for reuse elsewhere."""
    from kagome_vqe.results import load_optimal_params, load_run

    run = load_run(run_dir)
    theta = load_optimal_params(run_dir)
    if out is None:
        _print_json(
            {
                "run": run.name,
                "ansatz": run.config.ansatz.model_dump(mode="json"),
                "lattice": run.config.lattice.model_dump(mode="json", exclude_none=True),
                "num_parameters": int(theta.size),
                "final_energy": run.result.get("final_energy"),
                "optimal_params": theta.tolist(),
            }
        )
        return
    import numpy as np

    if out.suffix == ".npy":
        np.save(out, theta)
    else:
        np.savetxt(out, theta)
    console.print(f"wrote {theta.size} parameters to {out}")


# -- benchmark --------------------------------------------------------------------------------


@app.command()
def benchmark(
    config: Annotated[
        Path, typer.Argument(help="Benchmark config TOML.", exists=True, dir_okay=False)
    ] = Path("configs/benchmark.toml"),
    set_: SetOption = None,
    backend_override: BackendOverride = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Report the plan only.")] = False,
    point: Annotated[
        float | None,
        typer.Option(
            "--point",
            help="Evaluate every combo at fixed parameters (all equal to this value) instead of "
            "optimising; reports bias against the noise-free energy and spread between repeats.",
        ),
    ] = None,
    repeats: Annotated[int, typer.Option(help="Repeats per combo with --point.")] = 10,
    verbose: Verbose = False,
) -> None:
    """Run every (ansatz, optimiser) combo of CONFIG on one backend and tabulate."""
    from kagome_vqe.plotting import plot_ideal_vs_shots, plot_runs
    from kagome_vqe.results import load_run, write_benchmark_summary

    configure_logging(logging.DEBUG if verbose else logging.INFO)
    bench = BenchmarkConfig.model_validate(_load(config, set_, backend_override))
    if point is not None:
        _evaluate_benchmark(bench, point, repeats)
        return
    run_dirs: list[Path] = []
    for cfg in bench.run_configs():
        console.rule(cfg.name)
        run_dir = _run_one(cfg, dry_run=dry_run, progress=True)
        if run_dir is not None:
            run_dirs.append(run_dir)
    if dry_run:
        return
    runs = [load_run(d) for d in run_dirs]
    out_dir = Path(bench.output_dir) / f"{bench.name}-summary"
    json_path, md_path = write_benchmark_summary(runs, out_dir)
    fig_path = out_dir / "convergence.png"
    plot_runs(
        runs, axis="iteration", out=fig_path, title=f"{bench.name} on {bench.backend.label()}"
    )
    ideal_path = out_dir / "ideal_vs_shots.png"
    has_ideal = plot_ideal_vs_shots(
        runs, out=ideal_path, title=f"{bench.name}: parameter quality vs cost"
    )
    console.print(md_path.read_text())
    console.print(f"summary: {json_path}\nfigure:  {fig_path}")
    if has_ideal is not None:
        console.print(f"figure:  {ideal_path}")


def _evaluate_benchmark(bench: BenchmarkConfig, point: float, repeats: int) -> None:
    from kagome_vqe.results import write_evaluation_summary
    from kagome_vqe.vqe import evaluate_fixed

    summaries = []
    for cfg in bench.run_configs():
        console.rule(f"{cfg.name} at {point}")
        result = evaluate_fixed(cfg, point, repeats=repeats)
        summaries.append(result.summary())
        log.info(
            "%s: mean %.4f, spread %.4f, noise-free %.4f, bias %+.4f",
            cfg.name,
            result.mean,
            result.spread,
            result.ideal_energy,
            result.bias,
        )
    tag = f"{point:g}".replace(".", "p").replace("-", "m")
    out_dir = Path(bench.output_dir) / f"{bench.name}-evaluation-{tag}"
    json_path, md_path = write_evaluation_summary(summaries, out_dir)
    console.print(md_path.read_text())
    console.print(f"summary: {json_path}")


# -- study ------------------------------------------------------------------------------------


@app.command()
def study(
    config: Annotated[Path, typer.Argument(help="Study config TOML.", exists=True, dir_okay=False)],
    resume: Annotated[
        Path | None,
        typer.Option(
            help="Continue this study directory (skips its completed runs).",
            exists=True,
            file_okay=False,
        ),
    ] = None,
    only: Annotated[
        list[str] | None,
        typer.Option(help="Only runs whose name contains this text (repeatable)."),
    ] = None,
    max_runs: Annotated[
        int | None, typer.Option(help="Start at most this many runs, then summarise.")
    ] = None,
    time_limit: Annotated[
        float | None, typer.Option(help="Start no new run after this many minutes.")
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the run plan and a timed cost estimate only."),
    ] = False,
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Parent of the new study directory (default: output_dir in CONFIG)."),
    ] = None,
    verbose: Verbose = False,
) -> None:
    """Run a multi-seed experiment matrix (lattice x p x H0 x backend x optimiser x seed).

    Every run is saved as an ordinary results directory under <study>/runs/; the study's
    summary (tables, report.md, figures) is written to <study>/summary/. Interrupted
    studies continue with --resume.
    """
    from kagome_vqe.study import estimate_study, load_study_config, run_study

    configure_logging(logging.DEBUG if verbose else logging.INFO)
    cfg = load_study_config(config)
    if dry_run:
        frame, totals = estimate_study(cfg)
        cols = [
            "name",
            "num_parameters",
            "maxiter",
            "evaluation_budget",
            "shots_budget",
            "seconds_per_evaluation",
            "estimated_seconds",
        ]
        console.print(frame[cols].to_string(index=False))
        _print_json(totals)
        return
    study_dir = run_study(
        cfg,
        study_dir=resume,
        root=output_dir,
        only=only,
        max_runs=max_runs,
        time_limit_s=None if time_limit is None else 60 * time_limit,
    )
    console.print(f"study directory: {study_dir}")
    console.print(f"report: {study_dir / 'summary' / 'report.md'}")


@app.command(name="study-summary")
def study_summary(
    study_dir: Annotated[
        Path, typer.Argument(help="A study directory.", exists=True, file_okay=False)
    ],
) -> None:
    """Regenerate a study's summary tables, report and figures from its saved runs."""
    from kagome_vqe.study import summarise_study

    configure_logging(logging.INFO)
    console.print(f"wrote {summarise_study(study_dir)}")


# -- plot -------------------------------------------------------------------------------------


@app.command()
def plot(
    results: Annotated[list[Path], typer.Argument(help="Result directories.", exists=True)],
    out: Annotated[Path, typer.Option(help="Output image path.")] = Path("convergence.png"),
    axis: Annotated[str, typer.Option(help="'iteration' or 'evaluation'.")] = "iteration",
    title: Annotated[str | None, typer.Option(help="Figure title.")] = None,
    kind: Annotated[
        str, typer.Option(help="'convergence' (measured energy) or 'ideal' (noise-free vs shots).")
    ] = "convergence",
) -> None:
    """Plot saved runs: measured energy per iteration, or noise-free error against shots."""
    from kagome_vqe.plotting import plot_ideal_vs_shots, plot_runs
    from kagome_vqe.results import load_run

    if axis not in ("iteration", "evaluation"):
        msg = "--axis must be 'iteration' or 'evaluation'"
        raise typer.BadParameter(msg)
    if kind not in ("convergence", "ideal"):
        msg = "--kind must be 'convergence' or 'ideal'"
        raise typer.BadParameter(msg)
    runs = [load_run(p) for p in results]
    if kind == "ideal":
        if plot_ideal_vs_shots(runs, out=out, title=title) is None:
            msg = "none of these runs recorded a noise-free trace"
            raise typer.BadParameter(msg)
    else:
        plot_runs(runs, axis=axis, out=out, title=title)  # type: ignore[arg-type]
    console.print(f"wrote {out}")


# -- backends ---------------------------------------------------------------------------------


@app.command()
def backends(
    fake: Annotated[bool, typer.Option("--fake", help="List fake backends.")] = True,
    ibm: Annotated[bool, typer.Option("--ibm", help="Also query IBM Quantum.")] = False,
    min_qubits: Annotated[int, typer.Option(help="Hide smaller devices.")] = 12,
) -> None:
    """List fake backends bundled with qiskit-ibm-runtime and, optionally, your IBM devices."""
    from kagome_vqe.backends import ibm_service, list_fake_backends

    configure_logging(logging.WARNING)
    if fake:
        table = Table(title=f"Fake backends with >= {min_qubits} qubits")
        table.add_column("config name")
        table.add_column("device")
        table.add_column("qubits", justify="right")
        for info in list_fake_backends(min_qubits):
            table.add_row(info.name, info.backend_name, str(info.num_qubits))
        console.print(table)
    if ibm:
        settings = Settings()
        service: Any = ibm_service(settings)
        table = Table(title="IBM Quantum backends")
        table.add_column("name")
        table.add_column("qubits", justify="right")
        table.add_column("status")
        table.add_column("queue", justify="right")
        for backend in service.backends(min_num_qubits=min_qubits):
            try:
                status = backend.status()
                state = "online" if status.operational else "offline"
                queue = str(status.pending_jobs)
            except Exception:
                state, queue = "?", "?"
            table.add_row(backend.name, str(backend.num_qubits), state, queue)
        console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()
