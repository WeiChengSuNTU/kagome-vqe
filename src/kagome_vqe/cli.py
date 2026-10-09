"""The ``kagome-vqe`` command line.

Commands::

    kagome-vqe exact                      exact spectrum of the kagome Hamiltonian
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
from typing import Annotated, Any

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
from kagome_vqe.logging import attach_file_handler, configure_logging

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
    t: Annotated[float, typer.Option(help="Exchange coupling.")] = 1.0,
) -> None:
    """Exact diagonalisation of the 12-site kagome Heisenberg Hamiltonian."""
    import numpy as np

    from kagome_vqe.hamiltonian import exact_spectrum, heisenberg_hamiltonian
    from kagome_vqe.lattice import kagome_unit_cell, perfect_matchings

    lattice = kagome_unit_cell(t)
    hamiltonian = heisenberg_hamiltonian(lattice)
    spectrum = exact_spectrum(hamiltonian, k=k)
    degeneracy = int(np.sum(np.isclose(spectrum, spectrum[0], atol=1e-8)))
    table = Table(title="Kagome unit cell, H = t * sum_<ij> (XX + YY + ZZ)")
    table.add_column("quantity")
    table.add_column("value")
    table.add_row("sites / bonds", f"{lattice.num_sites} / {lattice.num_bonds}")
    table.add_row("Pauli terms", str(len(hamiltonian)))
    table.add_row("lowest eigenvalues", ", ".join(f"{v:.8f}" for v in spectrum))
    table.add_row("ground-state degeneracy", str(degeneracy))
    table.add_row("dimer coverings", str(len(perfect_matchings(lattice))))
    table.add_row("dimer-product energy", f"{-3 * t * lattice.num_sites / 2:.4f}")
    console.print(table)


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
    from kagome_vqe.results import new_run_dir, save_run
    from kagome_vqe.vqe import TraceRow, prepare, run_vqe

    settings = Settings()
    if cfg.backend.kind == "ibm" and not settings.has_credentials:
        log.warning("IBM_QUANTUM_TOKEN not set; relying on credentials saved by qiskit-ibm-runtime")

    prepared = prepare(cfg, settings=settings)
    if dry_run:
        _print_json(prepared.summary())
        return None

    run_dir = new_run_dir(cfg)
    handler = attach_file_handler(run_dir / "run.log")
    try:
        total = prepared.projected_evaluations
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

            result = run_vqe(cfg, prepared=prepared, settings=settings, on_evaluation=on_eval)
        save_run(result, run_dir)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()

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
    table.add_row("trace minimum (biased low)", f"{result.best_energy:.6f}")
    table.add_row("evaluations / iterations", f"{result.nfev} / {result.nit}")
    table.add_row("total shots", f"{result.total_shots:,}")
    table.add_row("wall time", f"{result.wall_time_s:.1f} s")
    table.add_row("saved to", str(run_dir))
    console.print(table)
    return run_dir


# -- benchmark --------------------------------------------------------------------------------


@app.command()
def benchmark(
    config: Annotated[
        Path, typer.Argument(help="Benchmark config TOML.", exists=True, dir_okay=False)
    ] = Path("configs/benchmark.toml"),
    set_: SetOption = None,
    backend_override: BackendOverride = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Report the plan only.")] = False,
    verbose: Verbose = False,
) -> None:
    """Run every (ansatz, optimiser) combo of CONFIG on one backend and tabulate."""
    from kagome_vqe.plotting import plot_runs
    from kagome_vqe.results import load_run, write_benchmark_summary

    configure_logging(logging.DEBUG if verbose else logging.INFO)
    bench = BenchmarkConfig.model_validate(_load(config, set_, backend_override))
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
    console.print(md_path.read_text())
    console.print(f"summary: {json_path}\nfigure:  {fig_path}")


# -- plot -------------------------------------------------------------------------------------


@app.command()
def plot(
    results: Annotated[list[Path], typer.Argument(help="Result directories.", exists=True)],
    out: Annotated[Path, typer.Option(help="Output image path.")] = Path("convergence.png"),
    axis: Annotated[str, typer.Option(help="'iteration' or 'evaluation'.")] = "iteration",
    title: Annotated[str | None, typer.Option(help="Figure title.")] = None,
) -> None:
    """Plot energy-versus-iteration for one or more saved runs."""
    from kagome_vqe.plotting import plot_runs
    from kagome_vqe.results import load_run

    if axis not in ("iteration", "evaluation"):
        msg = "--axis must be 'iteration' or 'evaluation'"
        raise typer.BadParameter(msg)
    runs = [load_run(p) for p in results]
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
