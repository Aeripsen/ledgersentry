#!/usr/bin/env python
"""CLI entry point: `python scripts/bootstrap.py`.

Thin wrapper so the bootstrap runs without an editable install. The pipeline
lives in src/ledgersentry/bootstrap.py (also runnable as `python -m
ledgersentry.bootstrap` once the package is installed).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ledgersentry.bootstrap import main  # noqa: E402

if __name__ == "__main__":
    main()
