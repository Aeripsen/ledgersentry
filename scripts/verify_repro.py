#!/usr/bin/env python
"""Verify the real-data run reproduces the committed numbers exactly.

Run AFTER `python scripts/train.py` with data/creditcard.csv present:

    python scripts/train.py && python scripts/verify_repro.py

Two checks, strictest first:
  1. Byte-identical: the regenerated artifacts/metrics_ulb_creditcard.json must
     match the committed copy (via `git show HEAD:...`). This is the real
     reproducibility claim - same data, same pinned deps, same file, byte for byte.
  2. Headline values: PR-AUC, recall at full coverage, and fold sizes must match
     the published numbers (a fallback check that still works outside a git clone).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
METRICS = REPO / "artifacts" / "metrics_ulb_creditcard.json"

EXPECTED = {
    "data_source": "ulb_creditcard",
    "is_synthetic": False,
    "n_train": 227846,
    "n_test": 56961,
    "n_test_fraud": 75,
    "pr_auc": 0.7278,
    "pr_auc_random_baseline": 0.0013,
    "recall_at_full_coverage": 0.84,
}


def main() -> int:
    if not METRICS.exists():
        print(f"FAIL: {METRICS} not found - run `python scripts/train.py` first")
        return 1
    regenerated = METRICS.read_bytes()

    # 1. byte-identical against the committed copy, when git is available
    try:
        committed = subprocess.run(
            ["git", "show", "HEAD:artifacts/metrics_ulb_creditcard.json"],
            cwd=REPO, capture_output=True, check=True,
        ).stdout
        if regenerated.replace(b"\r\n", b"\n") == committed.replace(b"\r\n", b"\n"):
            print("PASS: regenerated metrics are byte-identical to the committed file")
        else:
            print("FAIL: regenerated metrics differ from the committed file")
            return 1
    except (OSError, subprocess.CalledProcessError):
        print("note: git unavailable or file not committed - skipping byte check")

    # 2. headline values
    got = json.loads(regenerated)
    bad = {k: (got.get(k), v) for k, v in EXPECTED.items() if got.get(k) != v}
    if bad:
        for k, (actual, expected) in bad.items():
            print(f"FAIL: {k} = {actual!r}, expected {expected!r}")
        return 1
    print(f"PASS: PR-AUC {got['pr_auc']} / recall {got['recall_at_full_coverage']} "
          f"on {got['n_test']} held-out rows ({got['n_test_fraud']} fraud) all match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
