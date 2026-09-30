import numpy as np
import pytest

from ledgersentry.business import (
    bootstrap_comparison,
    compare_policies,
    masks_review_band,
    masks_single_threshold,
    policy_outcome,
    sensitivity_table,
)
from ledgersentry.model import curve_from_scores


def _fold(seed=0, n=5000):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.02).astype(int)
    p = np.clip(y * 0.6 + rng.normal(0.15, 0.2, size=n), 0, 1)
    amount = np.round(rng.lognormal(3.5, 1.2, size=n), 2)
    return p, y, amount


def test_every_fraud_lands_in_exactly_one_bucket():
    p, y, amount = _fold()
    for t in (0.5, 0.7, 0.9, 0.95):
        c = policy_outcome(*masks_review_band(p, t), y, amount, hours=5.0)["counts"]
        assert c["fraud_caught_auto"] + c["fraud_in_review"] + c["fraud_missed"] == y.sum()
        assert c["true_alerts"] + c["false_alerts"] == c["alerts_auto_flagged"]


def test_amount_partition_adds_up():
    p, y, amount = _fold()
    a = policy_outcome(*masks_review_band(p, 0.9), y, amount, hours=5.0)["amount"]
    parts = a["fraud_amount_caught_auto"] + a["fraud_amount_in_review"] + a["fraud_amount_missed"]
    assert parts == pytest.approx(a["fraud_amount_total"], abs=0.05)


def test_review_band_counts_match_the_shipped_curve():
    """The business table must describe the same knob the service ships."""
    p, y, amount = _fold()
    ts = (0.5, 0.6, 0.8, 0.95)
    curve = curve_from_scores(p, y, ts)
    for row, t in zip(curve, ts, strict=True):
        c = policy_outcome(*masks_review_band(p, t), y, amount, hours=1.0)["counts"]
        assert c["sent_to_review"] == row["n_sent_to_review"]
        assert c["alerts_auto_flagged"] == row["n_flagged_fraud"]
        assert c["fraud_caught_auto"] == row["fraud_caught_auto"]
        assert c["fraud_in_review"] == row["fraud_in_review_queue"]
        assert c["fraud_missed"] == row["fraud_missed"]


def test_single_threshold_has_no_review_queue():
    p, y, amount = _fold()
    c = policy_outcome(*masks_single_threshold(p, 0.5), y, amount, hours=1.0)["counts"]
    assert c["sent_to_review"] == 0 and c["fraud_in_review"] == 0


def test_flagged_and_review_must_be_disjoint():
    p, y, amount = _fold(n=10)
    both = np.ones(10, dtype=bool)
    with pytest.raises(ValueError):
        policy_outcome(both, both, y, amount, hours=1.0)


def test_band_raises_the_same_alerts_as_the_high_single_threshold():
    """B and C differ only in what happens to the uncertain middle."""
    p, y, amount = _fold()
    cmp = compare_policies(p, y, amount, hours=5.0, t=0.9)
    assert cmp["C_vs_B"]["same_alerts"]
    b_missed, c_missed = cmp["C_vs_B"]["fraud_missed"]
    assert c_missed <= b_missed


def test_surfaced_set_is_a_single_low_threshold():
    """The identity the module docstring states: auto-flag plus review is
    exactly p > 1 - t. The knob splits that set; it does not enlarge it."""
    p, y, amount = _fold()
    assert compare_policies(p, y, amount, hours=5.0, t=0.9)[
        "surfaced_equals_single_low_threshold"
    ]


def test_per_10k_rates_scale_with_counts():
    p, y, amount = _fold(n=20000)
    o = policy_outcome(*masks_review_band(p, 0.8), y, amount, hours=2.0)
    assert o["per_10k_transactions"]["review_queue"] == pytest.approx(
        o["counts"]["sent_to_review"] / 2, abs=0.05
    )
    assert o["per_hour_of_fold"]["review_queue"] == pytest.approx(
        o["counts"]["sent_to_review"] / 2, abs=0.05
    )


def test_nothing_is_priced():
    """No cost field may appear anywhere in the outcome: amounts are sums of the
    data, never a count multiplied by an assumed price."""
    p, y, amount = _fold()
    o = policy_outcome(*masks_review_band(p, 0.9), y, amount, hours=1.0)
    flat = repr(o)
    assert "cost" not in flat and "saving" not in flat


def test_bootstrap_is_seeded_and_brackets_the_point():
    p, y, amount = _fold(n=8000)
    b1 = bootstrap_comparison(p, y, amount, 0.9, n_resamples=200, seed=7)
    b2 = bootstrap_comparison(p, y, amount, 0.9, n_resamples=200, seed=7)
    assert b1 == b2
    point = compare_policies(p, y, amount, hours=1.0, t=0.9)["C_vs_A"][
        "false_alert_reduction_share"
    ]
    ci = b1["intervals_95"]["false_alert_reduction_share"]
    assert ci["ci_lower"] <= point <= ci["ci_upper"]


def test_sensitivity_review_load_is_monotone():
    p, y, amount = _fold()
    rows = sensitivity_table(p, y, amount, hours=1.0, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9))
    loads = [r["review_queue_per_10k"] for r in rows]
    assert loads == sorted(loads)
    assert rows[0]["review_queue_per_10k"] == 0
