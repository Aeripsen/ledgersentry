#!/usr/bin/env python
"""CLI entry point: `python scripts/headline.py`.

Thin wrapper so the headline check runs without an editable install. The logic
lives in src/ledgersentry/headline.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ledgersentry.headline import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
