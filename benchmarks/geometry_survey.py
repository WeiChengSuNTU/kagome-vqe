"""Exact low-energy data of the kagome clusters available in ``kagome_vqe``.

Usage::

    uv run python benchmarks/geometry_survey.py [--out CSV] [--max-sites 20]

For each cluster: sites, bonds, boundary, coordination numbers, corner-sharing triangles
(3-cliques) and complete hexagons (6-cycles of degree-4 sites around a kagome hole), and from
Lanczos on the full Hilbert space the ground energy, the energy per bond, the degeneracy of the
lowest level (within 1e-8) and the gap to the next distinct level, in units of ``J`` with
``H = sum (XX + YY + ZZ) = 4 sum S.S`` (the convention of every study here; divide by 4 for
``S.S`` units). No VQE: this is the physics reference for choosing clusters
(``docs/research_overview.md``, Sec. 1).
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from kagome_vqe.hamiltonian import exact_spectrum, heisenberg_hamiltonian
from kagome_vqe.kagome_lattices import kagome_bm_rectangle, kagome_kvqe20, kagome_periodic
from kagome_vqe.lattice import Lattice, kagome_unit_cell

CLUSTERS: dict[str, Callable[[], Lattice]] = {
    "star12 (IBM 2022 unit cell)": kagome_unit_cell,
    "bm2x5 (PRB 105 rectangle)": lambda: kagome_bm_rectangle(2, 5),
    "bm2x6 (PRB 105 rectangle)": lambda: kagome_bm_rectangle(2, 6),
    "kvqe20 (PRB 106 patch)": kagome_kvqe20,
    "torus 2x2 (12-site PBC)": lambda: kagome_periodic(2, 2),
    "torus 2x3 (18-site PBC)": lambda: kagome_periodic(2, 3),
}


def triangles(edges: list[tuple[int, int]]) -> int:
    """Number of 3-cliques (each is an elementary kagome triangle)."""
    adj: dict[int, set[int]] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return sum(
        1
        for a, b, c in itertools.combinations(sorted(adj), 3)
        if b in adj[a] and c in adj[a] and c in adj[b]
    )


def describe(name: str, lattice: Lattice) -> dict[str, object]:
    """Geometry and exact spectrum of one cluster."""
    edges = [(int(a), int(b)) for a, b, _ in lattice.edges]
    n = lattice.num_sites
    degree = Counter(x for e in edges for x in e)
    start = time.perf_counter()
    levels = np.sort(exact_spectrum(heisenberg_hamiltonian(lattice), k=8))
    e0 = float(levels[0])
    degenerate = int(np.sum(np.abs(levels - e0) < 1e-8))
    higher = levels[np.abs(levels - e0) >= 1e-8]
    gap = float(higher[0] - e0) if higher.size else float("nan")
    return {
        "cluster": name,
        "sites": n,
        "bonds": len(edges),
        "boundary": lattice.boundary,
        "bonds_per_site": round(len(edges) / n, 3),
        "degree_4_sites": sum(1 for v in range(n) if degree[v] == 4),
        "degree_lt4_sites": sum(1 for v in range(n) if degree[v] < 4),
        "triangles": triangles(edges),
        "E0": round(e0, 6),
        "E0_per_bond": round(e0 / len(edges), 4),
        "E0_per_site_SS": round(e0 / 4 / n, 4),
        "ground_degeneracy_in_lowest8": degenerate,
        "gap_to_next_level": round(gap, 4),
        "relative_gap": round(gap / abs(e0), 4),
        "lanczos_s": round(time.perf_counter() - start, 1),
    }


def main() -> None:
    """Tabulate every cluster up to ``--max-sites``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-sites", type=int, default=20)
    args = parser.parse_args()
    rows = []
    for name, build in CLUSTERS.items():
        lattice = build()
        if lattice.num_sites > args.max_sites:
            continue
        rows.append(describe(name, lattice))
        sys.stderr.write(f"{name}: done in {rows[-1]['lanczos_s']} s\n")
    table = pd.DataFrame(rows)
    if args.out:
        table.to_csv(args.out, index=False)
    sys.stdout.write(table.to_string(index=False) + "\n")


if __name__ == "__main__":
    main()
