import numpy as np
import pandas as pd
import pytest

from ledgersentry.business import (
    bootstrap_comparison,
    compare_policies,
    masks_review_band,
    masks_single_threshold,
    pct,
    policy_outcome,
    sensitivity_table,
    split_boundary,
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
        c = policy_outcome(*masks_review_band(p, t), y, amount)["counts"]
        assert c["fraud_caught_auto"] + c["fraud_in_review"] + c["fraud_missed"] == y.sum()
        assert c["true_alerts"] + c["false_alerts"] == c["alerts_auto_flagged"]
        assert c["legit_flagged_or_queued"] == c["false_alerts"] + c["legit_sent_to_review"]
        assert c["flagged_or_queued"] == c["alerts_auto_flagged"] + c["sent_to_review"]


def test_amount_partition_adds_up():
    p, y, amount = _fold()
    a = policy_outcome(*masks_review_band(p, 0.9), y, amount)["amount"]
    parts = a["fraud_amount_caught_auto"] + a["fraud_amount_in_review"] + a["fraud_amount_missed"]
    assert parts == pytest.approx(a["fraud_amount_total"], abs=0.05)


def test_amounts_are_sums_of_the_data_not_prices():
    """Scaling the Amount column scales every amount and leaves every share and
    count alone. A per-fraud price or a fixed cost term would break this."""
    p, y, amount = _fold()
    o1 = policy_outcome(*masks_review_band(p, 0.9), y, amount)
    o3 = policy_outcome(*masks_review_band(p, 0.9), y, amount * 3)
    assert o1["counts"] == o3["counts"]
    for k, v in o1["amount"].items():
        if "share" in k:
            assert o3["amount"][k] == v
        else:
            assert o3["amount"][k] == pytest.approx(3 * v, abs=0.05)


def test_review_band_counts_match_the_shipped_curve():
    """The business table must describe the same knob the service ships."""
    p, y, amount = _fold()
    ts = (0.5, 0.6, 0.8, 0.95)
    curve = curve_from_scores(p, y, ts)
    for row, t in zip(curve, ts, strict=True):
        c = policy_outcome(*masks_review_band(p, t), y, amount)["counts"]
        assert c["sent_to_review"] == row["n_sent_to_review"]
        assert c["alerts_auto_flagged"] == row["n_flagged_fraud"]
        assert c["fraud_caught_auto"] == row["fraud_caught_auto"]
        assert c["fraud_in_review"] == row["fraud_in_review_queue"]
        assert c["fraud_missed"] == row["fraud_missed"]


def test_single_threshold_has_no_review_queue():
    p, y, amount = _fold()
    c = policy_outcome(*masks_single_threshold(p, 0.5), y, amount)["counts"]
    assert c["sent_to_review"] == 0 and c["fraud_in_review"] == 0


def test_flagged_and_review_must_be_disjoint():
    p, y, amount = _fold(n=10)
    both = np.ones(10, dtype=bool)
    with pytest.raises(ValueError):
        policy_outcome(both, both, y, amount)


def test_false_alert_cut_belongs_to_the_threshold_not_the_queue():
    """B (threshold t, no review) and C (review band at t) raise the same alerts
    and the same false alerts, so the A-to-C cut in false alerts is exactly the
    A-to-B cut. Only the misses differ."""
    p, y, amount = _fold()
    cmp = compare_policies(p, y, amount, t=0.9)
    assert cmp["C_vs_B"]["same_alerts"] and cmp["C_vs_B"]["same_false_alerts"]
    assert (
        cmp["B_vs_A"]["false_alert_reduction_share"]
        == cmp["C_vs_A"]["false_alert_reduction_share"]
    )
    b_missed, c_missed = cmp["C_vs_B"]["fraud_missed"]
    assert c_missed <= b_missed
    assert cmp["C_vs_B"]["frauds_sent_to_review_instead_of_cleared"] == b_missed - c_missed


def test_surfaced_set_is_a_single_low_threshold():
    """Auto-flag plus review is exactly p > 1 - t (policy D). The knob splits
    that set; it does not enlarge it."""
    p, y, amount = _fold()
    cmp = compare_policies(p, y, amount, t=0.9)
    assert cmp["surfaced_equals_single_low_threshold"]
    c, d = cmp["C_review_band_0.9"], cmp["D_single_threshold_0.10"]
    assert c["counts"]["fraud_missed"] == d["counts"]["fraud_missed"]
    assert c["counts"]["flagged_or_queued"] == d["counts"]["alerts_auto_flagged"]


def test_per_10k_rates_scale_with_counts():
    p, y, amount = _fold(n=20000)
    o = policy_outcome(*masks_review_band(p, 0.8), y, amount)
    assert o["per_10k_transactions"]["review_queue"] == pytest.approx(
        o["counts"]["sent_to_review"] / 2, abs=0.05
    )


def test_bootstrap_is_seeded():
    p, y, amount = _fold(n=8000)
    b1 = bootstrap_comparison(p, y, amount, 0.9, n_resamples=200, seed=7)
    b2 = bootstrap_comparison(p, y, amount, 0.9, n_resamples=200, seed=7)
    b3 = bootstrap_comparison(p, y, amount, 0.9, n_resamples=200, seed=8)
    assert b1 == b2
    assert b1 != b3


def test_paired_interval_is_computed_within_each_resample():
    """Build a fold where A (0.5 cut) and C (band at 0.9) clear exactly the same
    frauds: every score is either <= 0.1 or >= 0.5. The paired difference is then
    zero in every resample, so its interval is [0, 0], while each policy's own
    miss share still varies from resample to resample. Two marginal intervals
    side by side could never give that."""
    rng = np.random.default_rng(3)
    n = 6000
    y = (rng.random(n) < 0.03).astype(int)
    p = np.where(rng.random(n) < 0.5, rng.uniform(0.0, 0.1, n), rng.uniform(0.5, 1.0, n))
    amount = np.round(rng.lognormal(3, 1, n), 2)
    ci = bootstrap_comparison(p, y, amount, 0.9, n_resamples=300, seed=1)["intervals_95"]
    assert ci["paired_A_minus_C_frauds_cleared_without_review"] == {
        "ci_lower": 0.0, "ci_upper": 0.0
    }
    assert ci["A_fraud_share_missed"]["ci_upper"] > ci["A_fraud_share_missed"]["ci_lower"]


def test_band_never_clears_a_fraud_the_default_cut_flags():
    """C clears p <= 1 - t, a subset of what the 0.5 cut clears (for t >= 0.5),
    so the paired A-minus-C miss difference is never negative."""
    p, y, amount = _fold(n=8000)
    ci = bootstrap_comparison(p, y, amount, 0.9, n_resamples=200, seed=7)["intervals_95"]
    assert ci["paired_A_minus_C_frauds_cleared_without_review"]["ci_lower"] >= 0


def test_split_boundary_reports_overlap_and_ties():
    ts = pd.to_datetime(["2013-01-01 00:00:0" + str(i) for i in range(6)])
    clean = split_boundary(pd.DataFrame({"timestamp": ts[:3]}), pd.DataFrame({"timestamp": ts[3:]}))
    assert clean["no_train_row_later_than_any_test_row"]
    assert clean["train_rows_at_boundary_second"] == 0
    tied = split_boundary(
        pd.DataFrame({"timestamp": ts[[0, 1, 3]]}), pd.DataFrame({"timestamp": ts[[3, 4]]})
    )
    assert tied["no_train_row_later_than_any_test_row"]
    assert tied["train_rows_at_boundary_second"] == 1
    leaky = split_boundary(
        pd.DataFrame({"timestamp": ts[[0, 5]]}), pd.DataFrame({"timestamp": ts[[2, 3]]})
    )
    assert not leaky["no_train_row_later_than_any_test_row"]
    assert leaky["train_rows_later_than_first_test_row"] == 1


def test_pct_never_rounds_to_zero_or_one_hundred():
    assert pct(0.9995) == "99.95%"
    assert pct(0.0004) == "0.04%"
    assert pct(1.0) == "100.0%"
    assert pct(0.972) == "97.2%"


def test_sensitivity_review_load_is_monotone():
    p, y, amount = _fold()
    rows = sensitivity_table(p, y, amount, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9))
    loads = [r["review_queue_per_10k"] for r in rows]
    assert loads == sorted(loads)
    assert rows[0]["review_queue_per_10k"] == 0
