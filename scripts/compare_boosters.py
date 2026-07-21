#!/usr/bin/env python
"""CLI entry point: `python scripts/compare_boosters.py`.

Thin wrapper so the LightGBM-vs-incumbent comparison runs without an editable
install. The pipeline lives in src/ledgersentry/compare_boosters.py (also
runnable as `python -m ledgersentry.compare_boosters` once the package is
installed). Needs the optional analysis install: `make install-analysis`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ledgersentry.compare_boosters import main  # noqa: E402

if __name__ == "__main__":
    main()
