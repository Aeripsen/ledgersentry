"""The live demo page cannot drift from the committed artifacts.

Runs offline in CI (no data/creditcard.csv needed). The committed
per-transaction export must rebuild the committed knob table and PR-AUC in
metrics_ulb_creditcard.json and the whole business_case_ulb_creditcard.json
(policies, 50-row sweep, bootstrap intervals), using business.py's own
functions. The page reads those same files, so if this passes, every number on
it traces to a committed artifact. pages.yml runs this before deploying."""
import json
from pathlib import Path

import pytest

from ledgersentry.demo import check_export

ART = Path(__file__).resolve().parents[1] / "artifacts"


def _load(name):
    path = ART / name
    if not path.exists():
        pytest.skip(f"{name} not committed")
    return json.loads(path.read_text())


def test_demo_export_rebuilds_metrics_and_business_case():
    demo = _load("demo_scores_ulb_creditcard.json")
    assert demo["is_synthetic"] is False
    errors = check_export(
        demo, _load("metrics_ulb_creditcard.json"), _load("business_case_ulb_creditcard.json")
    )
    assert errors == []


def test_readme_top_block_quotes_the_business_case():
    """The 30-second block is the first thing a reader sees, so it is held to
    the artifact the same way the page is."""
    b = _load("business_case_ulb_creditcard.json")
    readme = (ART.parent / "README.md").read_text(encoding="utf-8")
    t = b["operating_threshold"]
    pol = b["policies"]
    a, c = pol["A_single_threshold_0.5"], pol[f"C_review_band_{t}"]
    ci = b["bootstrap"]["intervals_95"]["false_alert_reduction_share"]
    red = pol["C_vs_A"]["false_alert_reduction_share"]
    review_share = c["counts"]["sent_to_review"] / c["counts"]["n_transactions"]
    expected = [
        f"**False fraud alerts fall {red:.0%}**, from "
        f"{a['per_10k_transactions']['false_alerts']:.1f} to "
        f"{c['per_10k_transactions']['false_alerts']:.1f} per 10,000 transactions "
        f"(95% CI {ci['ci_lower']:.0%} to {ci['ci_upper']:.0%}).",
        f"**Frauds approved silently fall from {a['counts']['fraud_missed']} to "
        f"{c['counts']['fraud_missed']} of {b['fold']['n_fraud']}**; the other "
        f"{c['counts']['fraud_in_review']} land in the review queue.",
        f"{round(c['per_10k_transactions']['review_queue'])} of every 10,000 transactions go "
        f"to a human reviewer ({review_share:.1%}).",
    ]
    for line in expected:
        assert line in readme, line
