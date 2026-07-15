#!/usr/bin/env python
"""CLI entry point: `python scripts/stream.py [--n N] [--review-threshold T]`.

Thin wrapper so the replay runs without an editable install. The actual logic
lives in src/ledgersentry/stream.py (also runnable as `python -m
ledgersentry.stream` once the package is installed).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ledgersentry.stream import main  # noqa: E402

if __name__ == "__main__":
    main()
