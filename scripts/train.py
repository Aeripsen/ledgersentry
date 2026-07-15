#!/usr/bin/env python
"""CLI entry point: `python scripts/train.py`.

Thin wrapper so training runs without an editable install. The actual pipeline
lives in src/ledgersentry/train.py (also runnable as `python -m ledgersentry.train`
once the package is installed).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ledgersentry.train import main  # noqa: E402

if __name__ == "__main__":
    main()
