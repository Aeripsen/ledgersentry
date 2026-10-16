"""Recompute the README's headline numbers from the committed held-out scores.

    python scripts/headline.py        (or: make headline)

Reads `artifacts/demo_scores_ulb_creditcard.json`, which holds one row per
transaction of the temporal test fold (the model's raw P(fraud), the label and
the dataset's own Amount), and recomputes from it, with this repo's own
functions: PR-AUC and its no-skill baseline, the reject-knob table, and the
four named policies of the business case. It prints them, then exits non-zero
unless every number equals the committed `metrics_ulb_creditcard.json` and
`business_case_ulb_creditcard.json` (via `demo.check_export`, the same check
the test suite runs). No download, no training, a few seconds.

What this proves: every published number follows from the committed scores, so
none of them was typed in by hand. What it does not prove: that those scores
came from the committed code. It never trains. `make reproduce` retrains on the
real data file (one public download) and requires the metrics back byte for
byte, and the ci `demo` job does the same on every push.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sklearn.metrics import average_precision_score

from .business import compare_policies
from .config import get_settings
from .demo import check_export, rows_from_export
from .model import curve_from_scores

SOURCE = "ulb_creditcard"
# The rows the README's knob table shows (the metrics file also holds 0.6, 0.99, 1.0).
README_THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95)


def load_artifacts(artifact_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """The committed export, metrics and business case. A missing file raises
    FileNotFoundError naming it, rather than silently checking less."""
    def read(name: str) -> dict[str, Any]:
        path = artifact_dir / name
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; it is committed in a clean clone")
        return json.loads(path.read_text(encoding="utf-8"))

    return (
        read(f"demo_scores_{SOURCE}.json"),
        read(f"metrics_{SOURCE}.json"),
        read(f"business_case_{SOURCE}.json"),
    )


def recompute(demo: dict[str, Any], operating_threshold: float) -> dict[str, Any]:
    """The headline numbers, rebuilt from the per-row export alone. Nothing in the
    result is read from the metrics or business files."""
    p, y, amount = rows_from_export(demo)
    n, n_fraud = len(p), int(y.sum())
    return {
        "n_test": n,
        "n_test_fraud": n_fraud,
        "fold_hours": demo["fold_hours"],
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "pr_auc_random_baseline": round(n_fraud / n, 4),
        "knob": curve_from_scores(p, y, list(README_THRESHOLDS)),
        "policies": compare_policies(p, y, amount, operating_threshold),
        "operating_threshold": operating_threshold,
    }


def format_report(r: dict[str, Any]) -> str:
    """Plain-text report of `recompute`'s output, in the README's own terms."""
    t = r["operating_threshold"]
    pol = r["policies"]
    a, b, c = (pol["A_single_threshold_0.5"], pol[f"B_single_threshold_{t}"],
               pol[f"C_review_band_{t}"])
    lines = [
        f"Temporal holdout: {r['n_test']:,} transactions, {r['n_test_fraud']} fraud, "
        f"last {r['fold_hours']} hours of the ULB set",
        f"PR-AUC {r['pr_auc']:.4f}  (no-skill baseline {r['pr_auc_random_baseline']:.4f})",
        "",
        "Reject-to-review knob (scores strictly between 1-t and t go to a human):",
        "  t      coverage  to review  flagged  precision  caught/review/missed  recall(auto)",
    ]
    for row in r["knob"]:
        lines.append(
            f"  {row['review_threshold']:<5}  {row['coverage']:>8.2%}"
            f"  {row['n_sent_to_review']:>9,}  {row['n_flagged_fraud']:>7}"
            f"  {row['precision_on_flagged']:>9.2%}"
            f"  {row['fraud_caught_auto']:>6} / {row['fraud_in_review_queue']:>2} / "
            f"{row['fraud_missed']:<2}         {row['recall_auto']:>6.1%}"
        )
    pa, pb, pc = (x["per_10k_transactions"] for x in (a, b, c))
    lines += [
        "",
        f"Per 10,000 transactions (A = cut at 0.5, B = cut at {t}, C = review band {t}):",
        f"  false alerts      A {pa['false_alerts']:.1f}   B {pb['false_alerts']:.1f}"
        f"   C {pc['false_alerts']:.1f}",
        f"  sent to review    A {pa['review_queue']:.1f}   B {pb['review_queue']:.1f}"
        f"   C {pc['review_queue']:.1f}",
        f"  frauds cleared unseen (of {c['counts']['fraud_total']}):  "
        f"A {a['counts']['fraud_missed']}   B {b['counts']['fraud_missed']}"
        f"   C {c['counts']['fraud_missed']}",
        f"  C: auto-flags {c['rates']['precision_on_alerts']:.1%} precise, "
        f"{c['counts']['fraud_caught_auto']} caught + {c['counts']['fraud_in_review']} in review"
        f" = {c['rates']['fraud_share_surfaced']:.0%} of frauds surfaced",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the recomputed headline and return 0 only if it matches the committed files."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    ap.add_argument("--artifacts", type=Path, default=get_settings().artifact_dir,
                    help="directory holding the committed artifacts (default: artifacts/)")
    args = ap.parse_args(argv)

    demo, metrics, business = load_artifacts(args.artifacts)
    print(format_report(recompute(demo, business["operating_threshold"])))
    errors = check_export(demo, metrics, business)
    print()
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1
    print("PASS: every number above, the full knob table, PR-AUC and the business case's "
          "policies, sweep, bootstrap intervals and sentences equal the committed artifacts.")
    print("This recomputes from committed scores; `make reproduce` retrains to prove the scores.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
