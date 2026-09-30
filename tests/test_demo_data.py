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
    the artifact the same way the page is, including which policy change each
    number is credited to."""
    b = _load("business_case_ulb_creditcard.json")
    readme = (ART.parent / "README.md").read_text(encoding="utf-8")
    t = b["operating_threshold"]
    pol = b["policies"]
    a, bb = pol["A_single_threshold_0.5"], pol[f"B_single_threshold_{t}"]
    c = pol[f"C_review_band_{t}"]
    assert pol["C_vs_B"]["same_false_alerts"], "the top block credits the cut to the threshold"
    ci = b["bootstrap"]["intervals_95"]
    red_ci = ci["false_alert_reduction_share"]
    miss_ci = ci["paired_A_minus_C_frauds_cleared_without_review"]
    red = pol["B_vs_A"]["false_alert_reduction_share"]
    review_share = c["counts"]["sent_to_review"] / c["counts"]["n_transactions"]
    expected = [
        f"**Raising the auto-flag bar from 0.5 to {t} cuts false fraud alerts {red:.0%}**, from "
        f"{a['per_10k_transactions']['false_alerts']:.1f} to "
        f"{bb['per_10k_transactions']['false_alerts']:.1f} per 10,000 transactions "
        f"(95% CI {red_ci['ci_lower']:.0%} to {red_ci['ci_upper']:.0%}). That is the "
        f"threshold's doing: the same bar with no review queue raises the identical "
        f"{bb['counts']['alerts_auto_flagged']} alerts.",
        f"alone, the {t} bar clears {bb['counts']['fraud_missed']} of {b['fold']['n_fraud']} "
        f"frauds unseen. Sending the uncertain middle to review cuts that to "
        f"{c['counts']['fraud_missed']}, against {a['counts']['fraud_missed']} at 0.5 "
        f"(paired 95% CI {miss_ci['ci_lower']:.0f} to {miss_ci['ci_upper']:.0f} fewer).",
        f"{round(c['per_10k_transactions']['review_queue'])} of every 10,000 transactions go "
        f"to a human reviewer ({review_share:.1%}), and {c['counts']['fraud_in_review']} of "
        f"those {c['counts']['sent_to_review']:,} are fraud.",
    ]
    for line in expected:
        assert line in readme, line


def test_readme_quotes_the_generated_resume_lines():
    """The resume lines are written by business.py; the README must quote them
    word for word, so a hand-edited resume claim cannot sit in the README."""
    b = _load("business_case_ulb_creditcard.json")
    readme = (ART.parent / "README.md").read_text(encoding="utf-8")
    assert len(b["resume_sentences"]) == 2
    for line in b["resume_sentences"]:
        assert f"> {line}" in readme, line
