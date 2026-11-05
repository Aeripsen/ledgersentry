#!/usr/bin/env python
"""CLI entry point: `python scripts/rolling_origin.py [--verify]`.

Thin wrapper so the analysis runs without an editable install. The pipeline
lives in src/ledgersentry/rolling.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ledgersentry.rolling import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
