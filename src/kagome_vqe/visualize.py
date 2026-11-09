"""Draw lattices: sites, bonds, wrapped bonds, a dimer covering and a schedule colouring.

Pure presentation; nothing in the physics or circuit modules imports this.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from kagome_vqe.kagome_lattices import cartesian
from kagome_vqe.lattice import Lattice

if TYPE_CHECKING:
    from kagome_vqe.schedule import ExchangeSchedule

__all__ = ["plot_lattice"]


def _positions(lattice: Lattice) -> dict[int, tuple[float, float]]:
    if lattice.coords is not None:
        return {s: cartesian(c) for s, c in enumerate(lattice.coords)}
    import rustworkx as rx

    layout = rx.spring_layout(lattice.to_rustworkx(), seed=1)
    return {s: (float(layout[s][0]), float(layout[s][1])) for s in range(lattice.num_sites)}


def _draw_bond(
    ax: Any,
    p: tuple[float, float],
    q: tuple[float, float],
    colour: Any,
    *,
    wrapped: bool,
) -> None:
    """A solid bond, or two short dashed stubs for a bond across a periodic boundary."""
    if not wrapped:
        ax.plot([p[0], q[0]], [p[1], q[1]], color=colour, lw=2, zorder=1)
        return
    length = max(abs(q[0] - p[0]) + abs(q[1] - p[1]), 1e-9)
    for (x0, y0), (x1, y1) in ((p, q), (q, p)):
        dx, dy = 0.3 * (x1 - x0) / length, 0.3 * (y1 - y0) / length
        ax.plot([x0, x0 + dx], [y0, y0 + dy], ls="--", color=colour, lw=1.2)


def plot_lattice(
    lattice: Lattice,
    *,
    schedule: ExchangeSchedule | None = None,
    show_dimers: bool = True,
    out: Path | None = None,
    ax: Any | None = None,
    title: str | None = None,
) -> Any:
    """Plot ``lattice``; optionally colour bonds by schedule layer and highlight dimers.

    Wrapped (periodic) bonds are drawn as short dashed stubs, since their endpoints are far
    apart in the plane. Sites are labelled with their site index.

    Returns:
        The matplotlib axes.
    """
    import matplotlib

    if out is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    pos = _positions(lattice)
    layer_of: dict[frozenset[int], int] = {}
    if schedule is not None:
        for k, layer in enumerate(schedule.layers):
            for a, b in layer:
                layer_of[frozenset((a, b))] = k
    wrapped = {frozenset(p) for p in lattice.wrapped_bonds}
    cmap = plt.get_cmap("tab10")
    for a, b, _ in lattice.edges:
        key = frozenset((a, b))
        colour = cmap(layer_of[key] % 10) if key in layer_of else "0.55"
        _draw_bond(ax, pos[a], pos[b], colour, wrapped=key in wrapped)
    if show_dimers and lattice.dimers:
        for a, b in lattice.dimers:
            (xa, ya), (xb, yb) = pos[a], pos[b]
            ax.plot([xa, xb], [ya, yb], color="black", lw=6, alpha=0.25, zorder=0)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.scatter(xs, ys, s=160, color="white", edgecolor="black", zorder=2)
    for s, (x, y) in pos.items():
        ax.text(x, y, str(s), ha="center", va="center", fontsize=7, zorder=3)
    if schedule is not None:
        from matplotlib.lines import Line2D

        labels = schedule.labels or tuple(f"layer {k}" for k in range(schedule.num_layers))
        handles = [
            Line2D([], [], color=cmap(k % 10), lw=2, label=label) for k, label in enumerate(labels)
        ]
        ax.legend(handles=handles, fontsize=7, loc="best")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        title
        or f"{lattice.name}: {lattice.num_sites} sites, {lattice.num_bonds} bonds "
        f"({lattice.boundary})"
    )
    if out is not None:
        figure: Any = ax.figure
        figure.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(figure)
    return ax
