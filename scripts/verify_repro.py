#!/usr/bin/env python
"""Verify the real-data run reproduces the committed numbers exactly.

Run AFTER `python scripts/train.py` with data/creditcard.csv present:

    python scripts/train.py && python scripts/verify_repro.py

Two checks, strictest first:
  1. Byte-identical: the regenerated artifacts/metrics_ulb_creditcard.json must
     match the committed copy (via `git show HEAD:...`). This is the real
     reproducibility claim - same data, same pinned deps, same file, byte for byte.
  2. Published values: PR-AUC, recall at full coverage, fold sizes, and ADR 002's
     demoted evidence (ROC-AUC, the accuracy pair) must match the published
     numbers (a fallback check that still works outside a git clone). Demoted
     does not mean unpinned: if this repo prints a number anywhere, that number
     reproduces.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
METRICS = REPO / "artifacts" / "metrics_ulb_creditcard.json"

_MISSING = object()

# Every number this repo publishes anywhere must reproduce, not just the headline.
# Dotted keys index into nested blocks.
EXPECTED = {
    "data_source": "ulb_creditcard",
    "is_synthetic": False,
    "n_train": 227846,
    "n_test": 56961,
    "n_test_fraud": 75,
    "pr_auc": 0.7278,
    "pr_auc_random_baseline": 0.0013,
    "recall_at_full_coverage": 0.84,
    # ADR 002's evidence: demoted on purpose, pinned all the same. The accuracy
    # pair is the argument - the model is WORSE than always-predict-legit, which
    # catches zero fraud.
    "demoted_metrics.roc_auc": 0.974,
    "demoted_metrics.accuracy": 0.9971,
    "demoted_metrics.accuracy_always_predict_legit": 0.9987,
}


def _dig(payload: dict, path: str) -> object:
    """Look up a dotted key path, returning _MISSING rather than raising."""
    node: object = payload
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


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
    bad = {k: (_dig(got, k), v) for k, v in EXPECTED.items() if _dig(got, k) != v}
    if bad:
        for k, (actual, expected) in bad.items():
            shown = "<missing>" if actual is _MISSING else repr(actual)
            print(f"FAIL: {k} = {shown}, expected {expected!r}")
        return 1
    print(f"PASS: PR-AUC {got['pr_auc']} / recall {got['recall_at_full_coverage']} "
          f"on {got['n_test']} held-out rows ({got['n_test_fraud']} fraud) all match")

    # 3. the live demo's per-transaction export (make demo-data) must rebuild
    # these metrics and the committed business case, so the page is held to the
    # same contract as the headline.
    demo_path = REPO / "artifacts" / "demo_scores_ulb_creditcard.json"
    business_path = REPO / "artifacts" / "business_case_ulb_creditcard.json"
    if demo_path.exists():
        sys.path.insert(0, str(REPO / "src"))
        from ledgersentry.demo import check_export

        business = json.loads(business_path.read_text()) if business_path.exists() else None
        errors = check_export(json.loads(demo_path.read_text()), got, business)
        if errors:
            for e in errors:
                print(f"FAIL: {e}")
            return 1
        print("PASS: the demo export rebuilds these metrics and the business case")
    return 0


if __name__ == "__main__":
    sys.exit(main())
