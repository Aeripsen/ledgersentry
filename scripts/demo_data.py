#!/usr/bin/env python
"""CLI entry point: `python scripts/demo_data.py`.

Thin wrapper so the demo export runs without an editable install. The pipeline
lives in src/ledgersentry/demo.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ledgersentry.demo import main  # noqa: E402

if __name__ == "__main__":
    main(sys.argv[1:])
