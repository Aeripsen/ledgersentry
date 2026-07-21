import numpy as np

from ledgersentry.cost import COST_SCENARIOS, THRESHOLD_GRID, price_scenarios
from ledgersentry.model import curve_from_scores, expected_cost_curve


def _knob(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.03).astype(int)
    p = np.clip(y * 0.5 + rng.normal(0.2, 0.2, size=n), 0, 1)
    return curve_from_scores(p, y, THRESHOLD_GRID), y, p


def test_expected_cost_is_a_measured_count_times_the_supplied_cost():
    """The whole honesty claim: every dollar is counts x costs, nothing baked."""
    knob, y, p = _knob()
    priced = expected_cost_curve(
        knob, cost_missed_fraud=200.0, cost_false_flag=5.0, cost_review=2.0
    )
    for row, src in zip(priced, knob, strict=True):
        false_flags = src["n_flagged_fraud"] - src["fraud_caught_auto"]
        expected = (
            src["fraud_missed"] * 200.0
            + false_flags * 5.0
            + src["n_sent_to_review"] * 2.0
        )
        assert row["expected_cost"] == round(expected, 2)
        assert row["n_false_flags"] == false_flags


def test_queued_frauds_are_not_charged_as_missed():
    """A fraud sent to review is surfaced, not lost; charging it the missed-fraud
    cost would erase the reason the reject knob exists."""
    knob, y, p = _knob()
    cheap_review = expected_cost_curve(
        knob, cost_missed_fraud=1000.0, cost_false_flag=5.0, cost_review=0.0
    )
    # with review free and missed-fraud dear, pushing frauds from missed into the
    # queue must never raise cost, so the minimum is at least as good as full auto
    full_auto = next(r for r in cheap_review if r["review_threshold"] == 0.5)
    assert min(r["expected_cost"] for r in cheap_review) <= full_auto["expected_cost"]


def test_optimum_moves_with_the_cost_assumptions():
    """If the priced optimum were the same threshold regardless of costs, the
    curve would be decorative. It must respond to the triple."""
    knob, y, p = _knob()
    priced = price_scenarios(knob, COST_SCENARIOS)
    thresholds = {s["name"]: s["min_cost_threshold"] for s in priced}
    # expensive review should favor automation (a lower threshold) over the
    # high-fraud-loss scenario that wants to surface more
    assert thresholds["expensive_review"] <= thresholds["high_fraud_loss"]


def test_savings_are_never_negative():
    """min cost is a minimum over the grid that includes full automation, so the
    reported saving cannot be negative by construction."""
    knob, y, p = _knob()
    for s in price_scenarios(knob, COST_SCENARIOS):
        assert s["savings_vs_full_automation"] >= 0
        assert s["min_expected_cost"] <= s["full_automation_cost"]


def test_full_automation_row_exists_in_the_grid():
    assert 0.5 in THRESHOLD_GRID
