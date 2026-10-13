"""Figure, table and trace generator for the milestone report and slide deck.

Python computes every number and figure once; the LaTeX report (``report/``) and the reveal.js
deck (``slides/``) only draw what this package writes. Run ``uv run python -m docsgen``.

Outputs (all committed, so neither document needs ``results/`` to build):

* ``report/tables/numbers.tex`` and ``slides/data/numbers.json``: every quoted number.
* ``report/figures/*.pdf`` and ``slides/media/*.svg``: matplotlib figures.
* ``report/tables/*.tex``: result tables.
* ``slides/data/*.json``: stepper traces for the deck.
"""

from __future__ import annotations

__all__: list[str] = []
