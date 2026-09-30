#!/usr/bin/env python
"""CLI entry point: `python scripts/loadtest.py`.

Thin wrapper so the HTTP load test runs without an editable install. The
harness lives in src/ledgersentry/loadtest.py (also runnable as `python -m
ledgersentry.loadtest` once the package is installed).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ledgersentry.loadtest import main  # noqa: E402

if __name__ == "__main__":
    main()
