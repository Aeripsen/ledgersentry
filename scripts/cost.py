#!/usr/bin/env python
"""CLI entry point: `python scripts/cost.py`.

Thin wrapper so the expected-cost-curve pipeline runs without an editable
install. The pipeline lives in src/ledgersentry/cost.py (also runnable as
`python -m ledgersentry.cost` once the package is installed).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ledgersentry.cost import main  # noqa: E402

if __name__ == "__main__":
    main()
