#!/usr/bin/env python
"""CLI entry point: `python scripts/shap_report.py`.

Thin wrapper so the SHAP attribution report runs without an editable install.
The pipeline lives in src/ledgersentry/shap_report.py (also runnable as
`python -m ledgersentry.shap_report` once the package is installed). Needs the
optional analysis install (`make install-analysis`) and an existing
artifacts/ledgersentry.joblib - it explains the shipped model, it never trains.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ledgersentry.shap_report import main  # noqa: E402

if __name__ == "__main__":
    main()
