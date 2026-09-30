#!/usr/bin/env python
"""CLI entry point: `python scripts/business_case.py [--verify]`.

Thin wrapper so the business-outcome pipeline runs without an editable install.
The pipeline lives in src/ledgersentry/business.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ledgersentry.business import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
