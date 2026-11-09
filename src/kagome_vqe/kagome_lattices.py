"""Generators for finite kagome lattices and custom interaction graphs.

All kagome generators work in the integer *triangular-lattice coordinates* described in
:mod:`kagome_vqe.lattice`: site ``(r, c)`` exists unless both ``r`` and ``c`` are odd, and the
bonds are the triangular nearest neighbours ``(0, +-1)``, ``(+-1, 0)`` and ``+-(1, 1)``. Every
generator returns a validated :class:`~kagome_vqe.lattice.Lattice` whose sites are numbered in
row-major order of their coordinates unless stated otherwise, so the numbering is
deterministic and documented.

Shipped geometries
------------------
``kagome_unit_cell``
    The 12-site "Star of David" of the IBM Open Science Prize (re-exported from
    :mod:`kagome_vqe.lattice`; its legacy site labels are kept).
``kagome_patch``
    Any open patch cut from a ``rows x cols`` window of the triangular lattice.
``kagome_kvqe20``
    The 20-site open patch of Kattemolle & van Wezel, PRB 106, 214429 (2022), Fig. 2, with its
    dimer covering.
``kagome_bm_rectangle``
    The kagome graph that the Bosse & Montanaro square-lattice embedding (PRB 105, 094409
    (2022), Fig. 2) induces on a ``rows x cols`` rectangle of qubits; their Fig. 4 lattices are
    labelled ``2x4, 2x6, 2x8, 3x6, 2x10, 3x8`` in this sense.
``kagome_periodic``
    An ``l1 x l2`` torus of three-site unit cells (``3 l1 l2`` sites, ``6 l1 l2`` bonds).
``custom_lattice``
    An explicit graph, optionally with coordinates and a dimer covering.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Final

from kagome_vqe.lattice import (
    Boundary,
    Lattice,
    is_kagome_hole,
    is_kagome_neighbour,
    kagome_unit_cell,
)

__all__ = [
    "KVQE20_COORDS",
    "KVQE20_DIMER_COORDS",
    "NEIGHBOUR_OFFSETS",
    "bm_coord_to_grid",
    "bm_grid_to_coord",
    "cartesian",
    "custom_lattice",
    "kagome_bm_rectangle",
    "kagome_from_coords",
    "kagome_kvqe20",
    "kagome_patch",
    "kagome_periodic",
    "kagome_unit_cell",
]

NEIGHBOUR_OFFSETS: Final[tuple[tuple[int, int], ...]] = ((0, 1), (1, 0), (1, 1))
"""Half of the triangular nearest-neighbour offsets; the other half are their negatives."""

KVQE20_COORDS: Final[tuple[tuple[int, int], ...]] = tuple(
    (r, c) for r in range(5) for c in range(5) if (r, c) != (0, 4) and not (r % 2 and c % 2)
)
"""The 20 sites of PRB 106 Fig. 2 in grid coordinates (row, column), row-major.

The patch occupies a 5 x 5 grid without its top-right corner; the four positions with odd row
and odd column are the hexagon centres, which KVQE_G uses as swapping stations.
"""

KVQE20_DIMER_COORDS: Final[tuple[tuple[tuple[int, int], tuple[int, int]], ...]] = (
    ((0, 0), (0, 1)), ((0, 2), (0, 3)),
    ((1, 0), (2, 1)), ((1, 2), (2, 3)),
    ((2, 0), (3, 0)), ((2, 2), (3, 2)),
    ((1, 4), (2, 4)), ((3, 4), (4, 4)),
    ((4, 0), (4, 1)), ((4, 2), (4, 3)),
)  # fmt: skip
"""The initial singlet covering of PRB 106 Fig. 2 (top right), as coordinate pairs."""


def cartesian(coord: tuple[int, int]) -> tuple[float, float]:
    """Planar position of a triangular-lattice point (unit bond length), for plotting.

    The sheared axes are undone: ``x = c - r/2``, ``y = -r sqrt(3)/2``.
    """
    r, c = coord
    return (c - r / 2, -r * math.sqrt(3) / 2)


def _bonds_from_coords(
    coords: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    index = {coord: i for i, coord in enumerate(coords)}
    bonds: list[tuple[int, int]] = []
    for i, (r, c) in enumerate(coords):
        for dr, dc in NEIGHBOUR_OFFSETS:
            j = index.get((r + dr, c + dc))
            if j is not None:
                bonds.append((min(i, j), max(i, j)))
    return sorted(bonds)


def _pairs_from_coords(
    coords: Sequence[tuple[int, int]],
    pairs: Iterable[tuple[tuple[int, int], tuple[int, int]]],
) -> tuple[tuple[int, int], ...]:
    index = {coord: i for i, coord in enumerate(coords)}
    try:
        return tuple((index[p], index[q]) for p, q in pairs)
    except KeyError as exc:
        msg = f"dimer endpoint {exc.args[0]} is not a site of the patch"
        raise ValueError(msg) from None


def kagome_from_coords(
    coords: Sequence[tuple[int, int]],
    t: float = 1.0,
    *,
    name: str = "kagome_patch",
    dimers: Sequence[tuple[int, int]] | None = None,
    require_connected: bool = True,
) -> Lattice:
    """The open kagome patch on the given sites, with every nearest-neighbour bond.

    Args:
        coords: Triangular-lattice coordinates; site ``i`` gets ``coords[i]``. Holes
            (odd, odd) are rejected.
        t: Uniform coupling.
        name: Lattice label.
        dimers: Optional dimer covering in site labels.
        require_connected: Reject patches that fall apart into several pieces.

    Raises:
        ValueError: On holes, duplicates, an invalid covering or a disconnected patch.
    """
    coords = tuple((int(r), int(c)) for r, c in coords)
    bonds = _bonds_from_coords(coords)
    lattice = Lattice(
        num_sites=len(coords),
        edges=tuple((a, b, float(t)) for a, b in bonds),
        name=name,
        coords=coords,
        boundary="open",
        dimers=None if dimers is None else tuple(dimers),
    )
    if require_connected and not lattice.is_connected():
        msg = f"{name}: the patch is not connected"
        raise ValueError(msg)
    return lattice


def kagome_patch(
    rows: int,
    cols: int,
    t: float = 1.0,
    *,
    drop: Iterable[tuple[int, int]] = (),
    name: str | None = None,
) -> Lattice:
    """Open patch on all kagome sites of the window ``0 <= r < rows``, ``0 <= c < cols``.

    Args:
        rows: Window height in triangular-lattice rows.
        cols: Window width.
        t: Uniform coupling.
        drop: Coordinates to leave out (e.g. a corner).
        name: Label; defaults to ``"kagome_patch_<rows>x<cols>"``.
    """
    if rows < 1 or cols < 1:
        msg = f"patch window must be at least 1 x 1, got {rows} x {cols}"
        raise ValueError(msg)
    dropped = {(int(r), int(c)) for r, c in drop}
    coords = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if not is_kagome_hole((r, c)) and (r, c) not in dropped
    ]
    return kagome_from_coords(coords, t, name=name or f"kagome_patch_{rows}x{cols}")


def kagome_kvqe20(t: float = 1.0) -> Lattice:
    """The 20-site open patch simulated by KVQE_G in PRB 106, 214429 (2022), Fig. 2.

    30 bonds; sites numbered row-major in grid coordinates (:data:`KVQE20_COORDS`); the dimer
    covering is the paper's (:data:`KVQE20_DIMER_COORDS`).
    """
    dimers = _pairs_from_coords(KVQE20_COORDS, KVQE20_DIMER_COORDS)
    return kagome_from_coords(KVQE20_COORDS, t, name="kagome_kvqe20", dimers=dimers)


# -- Bosse-Montanaro square-lattice family ----------------------------------------------------

_BM_COLUMN_PHASE: Final[int] = 2
"""Chosen so that grid row 0 reads ``L L M L L M ...`` as in PRB 105 Fig. 2 (``L``: site of a
horizontal kagome line, ``M``: apex site between two lines)."""


def bm_grid_to_coord(row: int, col: int) -> tuple[int, int]:
    """Triangular coordinate of the kagome site at square-grid position ``(row, col)``.

    Inverse of the Bosse-Montanaro embedding used by :mod:`kagome_vqe.mapping.prb105`: grid row
    ``g`` holds the horizontal kagome line ``r = 2 g`` with the apex sites of row ``r - 1``
    inserted between their two base sites. With ``J = col + 2 row + phase``, ``J = 0, 2 mod 3``
    are line sites and ``J = 1 mod 3`` apex sites. Every grid position hosts exactly one site.
    """
    big_j = col + 2 * row + _BM_COLUMN_PHASE
    residue = big_j % 3
    if residue == 0:
        return (2 * row, 2 * big_j // 3)
    if residue == 2:
        return (2 * row, (2 * big_j - 1) // 3)
    return (2 * row - 1, 2 * (big_j - 1) // 3)


def bm_coord_to_grid(coord: tuple[int, int]) -> tuple[int, int]:
    """Square-grid position ``(row, col)`` of a kagome site in the PRB 105 embedding.

    Exact inverse of :func:`bm_grid_to_coord`. Line sites ``(2g, c)`` go to row ``g``; apex
    sites ``(2g - 1, c)`` (``c`` even) go to row ``g`` between their two lower neighbours.

    Raises:
        ValueError: For a hexagon centre (odd, odd), which hosts no site.
    """
    r, c = coord
    if is_kagome_hole(coord):
        msg = f"{coord} is a hexagon centre, not a kagome site"
        raise ValueError(msg)
    if r % 2 == 0:
        row = r // 2
        big_j = c + (c + 1) // 2  # c + ceil(c / 2)
    else:
        row = (r + 1) // 2
        big_j = 3 * c // 2 + 1
    return (row, big_j - 2 * row - _BM_COLUMN_PHASE)


def kagome_bm_rectangle(rows: int, cols: int, t: float = 1.0) -> Lattice:
    """The kagome graph induced on a ``rows x cols`` qubit rectangle by the PRB 105 embedding.

    Sites are numbered row-major over the rectangle, so site ``g * cols + j`` sits on qubit
    ``(g, j)``. These are the lattices of Bosse & Montanaro's Fig. 4 (``2x4`` ... ``3x8``). The
    attached dimer covering is their initial state: the "colour 1" horizontal pairs
    ``(g, j)-(g, j+1)`` with even ``j``, which is a perfect matching only for even ``cols``
    (the paper restricts itself to those); for odd ``cols`` no covering is attached.

    Raises:
        ValueError: For an empty rectangle or a disconnected result.
    """
    if rows < 1 or cols < 2:
        msg = f"BM rectangle needs rows >= 1 and cols >= 2, got {rows} x {cols}"
        raise ValueError(msg)
    coords = [bm_grid_to_coord(g, j) for g in range(rows) for j in range(cols)]
    dimers = None
    if cols % 2 == 0:
        dimers = [(g * cols + j, g * cols + j + 1) for g in range(rows) for j in range(0, cols, 2)]
    return kagome_from_coords(coords, t, name=f"kagome_bm_{rows}x{cols}", dimers=dimers)


# -- periodic ---------------------------------------------------------------------------------


def kagome_periodic(l1: int, l2: int, t: float = 1.0) -> Lattice:
    """An ``l1 x l2`` torus of kagome unit cells (periodic in both directions).

    The unit cell holds sites ``(0, 0), (0, 1), (1, 0)`` and the lattice vectors are ``(2, 0)``
    and ``(0, 2)``. Bonds that leave the ``2 l1 x 2 l2`` window are wrapped and recorded in
    ``wrapped_bonds``. ``coords`` keep the unwrapped in-window positions (for plotting); the
    geometric mappings reject periodic lattices because wrapped bonds are not planar.

    Raises:
        ValueError: If ``l1`` or ``l2`` is below 2 (the wrap would create double bonds or
            self-loops).
    """
    if l1 < 2 or l2 < 2:
        msg = f"periodic kagome needs l1, l2 >= 2 to avoid double bonds, got {l1} x {l2}"
        raise ValueError(msg)
    height, width = 2 * l1, 2 * l2
    coords = [(r, c) for r in range(height) for c in range(width) if not is_kagome_hole((r, c))]
    index = {coord: i for i, coord in enumerate(coords)}
    bonds: set[tuple[int, int]] = set()
    wrapped: set[tuple[int, int]] = set()
    for i, (r, c) in enumerate(coords):
        for dr, dc in NEIGHBOUR_OFFSETS:
            target = ((r + dr) % height, (c + dc) % width)
            if is_kagome_hole(target):
                continue
            j = index[target]
            pair = (min(i, j), max(i, j))
            bonds.add(pair)
            if target != (r + dr, c + dc):
                wrapped.add(pair)
    lattice = Lattice(
        num_sites=len(coords),
        edges=tuple((a, b, float(t)) for a, b in sorted(bonds)),
        name=f"kagome_periodic_{l1}x{l2}",
        coords=tuple(coords),
        boundary="periodic",
        wrapped_bonds=tuple(sorted(wrapped)),
    )
    if lattice.num_bonds != 2 * lattice.num_sites:
        msg = f"torus {l1}x{l2} has {lattice.num_bonds} bonds, expected {2 * lattice.num_sites}"
        raise AssertionError(msg)
    return lattice


# -- custom -----------------------------------------------------------------------------------


def custom_lattice(
    num_sites: int,
    bonds: Sequence[Sequence[float]],
    *,
    coupling: float = 1.0,
    name: str = "custom",
    coords: Sequence[Sequence[int]] | None = None,
    boundary: Boundary = "unspecified",
    wrapped_bonds: Sequence[Sequence[int]] = (),
    dimers: Sequence[Sequence[int]] | None = None,
    require_connected: bool = True,
) -> Lattice:
    """An explicit interaction graph.

    No geometric assumption is made: the bond list alone defines the Hamiltonian and the
    graph-based ansatz. ``coords`` are only needed by the geometric hardware mappings, which
    additionally check that every bond is a kagome nearest neighbour in them.

    Args:
        num_sites: Number of sites, labelled ``0 .. num_sites - 1``.
        bonds: ``[a, b]`` (coupling ``coupling``) or ``[a, b, J_ab]`` per bond.
        coupling: Default coupling for two-element bonds.
        name: Label.
        coords: Optional triangular-lattice coordinates per site.
        boundary: Declared boundary condition.
        wrapped_bonds: Bonds crossing a periodic boundary (``boundary = "periodic"`` only).
        dimers: Optional dimer covering.
        require_connected: Reject disconnected graphs.

    Raises:
        ValueError: On malformed bonds, out-of-range sites, duplicates, zero couplings, an
            invalid covering, or a disconnected graph.
    """
    edges: list[tuple[int, int, float]] = []
    for bond in bonds:
        if len(bond) not in (2, 3):
            msg = f"bond {list(bond)} must be [a, b] or [a, b, J]"
            raise ValueError(msg)
        a, b = bond[0], bond[1]
        if int(a) != a or int(b) != b:
            msg = f"bond {list(bond)} has non-integer site labels"
            raise ValueError(msg)
        weight = float(bond[2]) if len(bond) == 3 else float(coupling)
        if not math.isfinite(weight):
            msg = f"bond {list(bond)} has a non-finite coupling"
            raise ValueError(msg)
        edges.append((int(a), int(b), weight))
    lattice = Lattice(
        num_sites=int(num_sites),
        edges=tuple(edges),
        name=name,
        coords=None if coords is None else tuple((int(p[0]), int(p[1])) for p in coords),
        boundary=boundary,
        wrapped_bonds=tuple((int(p[0]), int(p[1])) for p in wrapped_bonds),
        dimers=None if dimers is None else tuple((int(p[0]), int(p[1])) for p in dimers),
    )
    if require_connected and not lattice.is_connected():
        msg = f"{name}: the interaction graph is not connected"
        raise ValueError(msg)
    if lattice.coords is not None and lattice.boundary == "open":
        bad = [
            (a, b)
            for a, b, _ in lattice.edges
            if not is_kagome_neighbour(lattice.coords[a], lattice.coords[b])
        ]
        if bad:
            msg = (
                f"{name}: bonds {bad[:5]} are not kagome nearest neighbours in the given coords;"
                " fix the coordinates or omit them"
            )
            raise ValueError(msg)
    return lattice
