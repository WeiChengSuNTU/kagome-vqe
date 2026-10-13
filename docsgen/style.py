"""One palette for the report figures and the deck.

The first four tokens match ``slides/css/theme.css`` and keep one meaning everywhere:
accent = what matters now, derived = derived structure (fills, the ring), side-b = the second
family in a comparison, muted = background. ``METHOD_COLORS`` fixes one categorical hue per
gradient method (validated with the dataviz palette checker, light mode) so a method has the
same colour in every chart of both documents.
"""

from __future__ import annotations

from typing import Final

import matplotlib as mpl

__all__ = [
    "ACCENT",
    "ACCENT_INK",
    "DERIVED",
    "INK",
    "INK_2",
    "METHOD_COLORS",
    "METHOD_LABELS",
    "MUTED",
    "SIDE_B",
]

ACCENT: Final = "#d85a30"
ACCENT_INK: Final = "#993c1d"
DERIVED: Final = "#7f77dd"
SIDE_B: Final = "#1d9e75"
MUTED: Final = "#b4b2a9"
INK: Final = "#1f1e1c"
INK_2: Final = "#5f5e5a"

METHOD_COLORS: Final[dict[str, str]] = {
    "adam_forward": "#2a78d6",
    "adam_central": "#eb6834",
    "adam_spsa": "#1baf7a",
    "adam_parameter_shift": "#eda100",
    "spsa": "#e87ba4",
}
METHOD_LABELS: Final[dict[str, str]] = {
    "adam_forward": "ADAM, forward diff. (paper)",
    "adam_central": "ADAM, central diff.",
    "adam_spsa": "ADAM, SPSA gradient",
    "adam_parameter_shift": "ADAM, parameter shift",
    "spsa": "plain SPSA",
}


def apply() -> None:
    """Matplotlib defaults shared by every figure: thin marks, recessive grid, ink text."""
    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": INK_2,
            "axes.labelcolor": INK,
            "axes.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": MUTED,
            "grid.alpha": 0.35,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "savefig.bbox": "tight",
            "svg.fonttype": "none",
            "svg.hashsalt": "docsgen",  # deterministic clip-path ids, stable diffs
            "pdf.fonttype": 42,
        }
    )
