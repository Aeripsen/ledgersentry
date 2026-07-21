#!/usr/bin/env python
"""CLI entry point: `python scripts/compare.py`.

Thin wrapper so the feature-set / boosting-config comparison runs without an
editable install. The pipeline lives in src/ledgersentry/compare.py (also
runnable as `python -m ledgersentry.compare` once the package is installed).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ledgersentry.compare import main  # noqa: E402

if __name__ == "__main__":
    main()
