"""
The per-transaction export behind the live demo page
(https://aeripsen.github.io/ledgersentry/), and the check that keeps the page
from drifting away from the committed artifacts.

What it writes: artifacts/demo_scores_<source>.json, every held-out row of the
headline fold with the shipped model's raw P(fraud), the true label and the
dataset's own Amount. The page counts the knob's lanes from these rows in the
browser; it never runs the model and never invents a row.

Why it can be trusted: check_export() rebuilds, from the exported rows alone,
  1. the committed knob table and PR-AUC in metrics_<source>.json, and
  2. the whole policies / sweep / bootstrap block of business_case_<source>.json
     (business.py's own functions, called on these rows), and the headline and
     resume sentences built from them.
CI runs it offline through tests/test_demo_data.py before pages.yml deploys,
and scripts/verify_repro.py runs it after a real-data retrain. The business
numbers stay in ONE artifact (business.py); this module only ships the rows.

Scores are rounded to 8 decimals to keep the file under 1 MB. The check runs on
the rounded values, so rounding that flipped any count would fail it.

Run: python scripts/demo_data.py   (needs data/creditcard.csv; `make demo-data`)
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .business import (
    N_RESAMPLES,
    OPERATING_THRESHOLD,
    THRESHOLD_SWEEP,
    bootstrap_comparison,
    check_against_committed_metrics,
    compare_policies,
    headline_sentences,
    holdout_fold,
    resume_sentences,
    sensitivity_table,
)
from .config import get_settings
from .model import curve_from_scores

SCORE_DECIMALS = 8


def rows_from_export(demo: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(demo["p_fraud"], dtype=float)
    y = np.zeros(len(p), dtype=int)
    y[np.asarray(demo["fraud_rows"], dtype=int)] = 1
    return p, y, np.asarray(demo["amount"], dtype=float)


def check_export(
    demo: dict[str, Any], metrics: dict[str, Any], business: dict[str, Any] | None = None
) -> list[str]:
    """Mismatches between what the exported rows imply and what is committed.
    Empty list = every number the page shows traces to a committed artifact."""
    errors: list[str] = []
    p, y, amount = rows_from_export(demo)
    if len(p) != metrics["n_test"] or int(y.sum()) != metrics["n_test_fraud"]:
        errors.append("export fold size or fraud count differs from the metrics file")
    thresholds = [r["review_threshold"] for r in metrics["coverage_precision_curve"]]
    if curve_from_scores(p, y, thresholds) != metrics["coverage_precision_curve"]:
        errors.append("export does not regenerate the committed knob table")
    if round(float(average_precision_score(y, p)), 4) != metrics["pr_auc"]:
        errors.append("export does not regenerate the committed PR-AUC")
    if business is not None:
        t = business["operating_threshold"]
        seed = get_settings().random_state  # business.py seeds its bootstrap with this
        rebuilt: dict[str, Any] = {
            "policies": compare_policies(p, y, amount, t),
            "sensitivity_review_band": sensitivity_table(p, y, amount, THRESHOLD_SWEEP),
            "bootstrap": bootstrap_comparison(p, y, amount, t, N_RESAMPLES, seed),
        }
        # JSON round trip so tuples/np scalars compare the way the file stores them
        rebuilt = json.loads(json.dumps(rebuilt))
        # the sentences are a function of those blocks plus the split boundary,
        # which needs the training rows and so is taken from the committed file
        merged = {**business, **rebuilt}
        rebuilt["headline_sentences"] = headline_sentences(merged)
        rebuilt["resume_sentences"] = resume_sentences(merged)
        for key, value in rebuilt.items():
            if business.get(key) != value:
                errors.append(f"business_case[{key!r}] does not regenerate from the export")
    return errors


def main() -> dict[str, Any]:
    cfg = get_settings()
    source, test_df, p_raw = holdout_fold(cfg)
    p = np.round(p_raw, SCORE_DECIMALS)
    y = test_df["is_fraud"].to_numpy().astype(int)
    print(f"[check] {check_against_committed_metrics(cfg, source, p, y)}")
    ts = pd.to_datetime(test_df["timestamp"])
    hours = float((ts.max() - ts.min()).total_seconds() / 3600)
    metrics = json.loads((cfg.artifact_dir / f"metrics_{source}.json").read_text())

    demo = {
        "data_source": source,
        "is_synthetic": source == "synthetic",
        "what": (
            "Every held-out transaction of the temporal test fold: the shipped model's "
            "raw P(fraud) (8 decimals), the true label, and the dataset's own Amount "
            "(ULB states no currency). The demo page counts the knob's lanes from these "
            "rows; tests/test_demo_data.py rebuilds the committed metrics and business "
            "case from them."
        ),
        "rebuild": "make reproduce && make demo-data (needs data/creditcard.csv)",
        "n": int(len(p)),
        "fold_hours": round(hours, 2),
        "fold_hours_exact": hours,
        "operating_threshold": OPERATING_THRESHOLD,
        "committed_thresholds": [
            r["review_threshold"] for r in metrics["coverage_precision_curve"]
        ],
        "fraud_rows": [int(i) for i in np.flatnonzero(y == 1)],
        "p_fraud": [float(v) for v in p],
        "amount": [round(float(a), 2) for a in test_df["amount"].to_numpy(dtype=float)],
    }
    business_path = cfg.artifact_dir / f"business_case_{source}.json"
    business = json.loads(business_path.read_text()) if business_path.exists() else None
    errors = check_export(demo, metrics, business)
    if errors:
        raise SystemExit("FAIL: " + "; ".join(errors))
    out = cfg.artifact_dir / f"demo_scores_{source}.json"
    out.write_text(json.dumps(demo, separators=(",", ":")), newline="\n")
    print(f"[save ] {out} ({out.stat().st_size / 1e6:.2f} MB), rebuilds metrics + business case")
    return demo


if __name__ == "__main__":
    main()
