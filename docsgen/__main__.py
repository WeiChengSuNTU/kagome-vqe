"""``uv run python -m docsgen [--recompute] [--only NAME ...]``: regenerate every artefact."""

from __future__ import annotations

import argparse
import logging

from docsgen import figures, numbers, tables, traces


def main() -> None:
    """Numbers first (tables and slides read them), then tables, traces and figures."""
    parser = argparse.ArgumentParser(prog="python -m docsgen", description=__doc__)
    parser.add_argument(
        "--recompute", action="store_true", help="rerun the cached simulator studies"
    )
    parser.add_argument("--only", nargs="*", help="figure names to render (default: all)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    reg = numbers.collect(recompute=args.recompute)
    written = [numbers.write_tex(reg), numbers.write_json(reg)]
    written += tables.write_all(reg)
    written += traces.write_all()
    written += figures.write_all(set(args.only) if args.only else None)
    for path in written:
        print(f"wrote {path.relative_to(numbers.ROOT)}")  # noqa: T201


if __name__ == "__main__":
    main()
