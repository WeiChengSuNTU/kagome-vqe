"""Fixed drawing coordinates for the kagome star, shared by the report figures and the deck.

Hexagon sites 0-5 sit on a unit circle; tip ``6 + i`` caps hexagon edge ``(i, i+1)`` as an
equilateral triangle pointing outwards. The bond list itself always comes from
:data:`kagome_vqe.lattice.KAGOME_UNIT_CELL_EDGES`, never from here.
"""

from __future__ import annotations

import math

from kagome_vqe.lattice import KAGOME_UNIT_CELL_EDGES

__all__ = ["bond_kind", "positions", "triangles"]


def positions(scale: float = 1.0) -> dict[int, tuple[float, float]]:
    """Site -> (x, y), y pointing up. Hexagon radius ``scale``; tips at ``sqrt(3) * scale``."""
    pos: dict[int, tuple[float, float]] = {}
    for i in range(6):
        angle = math.pi / 2 - i * math.pi / 3  # site 0 at the top, clockwise
        pos[i] = (scale * math.cos(angle), scale * math.sin(angle))
    for i in range(6):
        angle = math.pi / 2 - (i + 0.5) * math.pi / 3
        r = math.sqrt(3) * scale
        pos[6 + i] = (r * math.cos(angle), r * math.sin(angle))
    return pos


def triangles() -> list[tuple[int, int, int]]:
    """The six corner-sharing triangles ``(i, i+1, tip)``; they partition the 18 bonds."""
    return [(i, (i + 1) % 6, 6 + i) for i in range(6)]


def bond_kind(a: int, b: int) -> str:
    """``"hexagon"`` for bonds between hexagon sites, ``"spoke"`` for bonds to a tip."""
    if frozenset((a, b)) not in {frozenset(e) for e in KAGOME_UNIT_CELL_EDGES}:
        msg = f"({a}, {b}) is not a bond of the kagome star"
        raise ValueError(msg)
    return "hexagon" if a < 6 and b < 6 else "spoke"
