#!/usr/bin/env python
"""CLI entry point: `python scripts/bench.py`.

Thin wrapper so the benchmark runs without an editable install. The harness
lives in src/ledgersentry/bench.py (also runnable as `python -m
ledgersentry.bench` once the package is installed).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ledgersentry.bench import main  # noqa: E402

if __name__ == "__main__":
    main()
