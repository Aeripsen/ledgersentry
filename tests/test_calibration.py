"""
Calibration is only trustworthy if its plumbing is: the cal slice must come
after the fit slice in time, calibrators must be monotone (they may fix the
probability SCALE, never the ranking), and the expected-cost math must be
checkable by hand.
"""
import numpy as np
import pytest
from sklearn.metrics import brier_score_loss

from ledgersentry import config, data
from ledgersentry.calibration import IsotonicCalibrator, PlattCalibrator
from ledgersentry.model import curve_from_scores, expected_cost_curve


def _scores_and_labels(n=4000, seed=9):
    """A deliberately OVERCONFIDENT score vector: true P(fraud) is milder than
    the score claims, the shape a boosted model fit on reweighted classes
    produces."""
    rng = np.random.default_rng(seed)
    p_true = rng.uniform(0.01, 0.99, size=n)
    y = (rng.random(n) < p_true).astype(int)
    # overconfident: push scores toward the extremes relative to p_true
    p_raw = np.clip(p_true + 0.35 * np.sign(p_true - 0.5) * p_true * (1 - p_true) * 4, 0, 1)
    return p_raw, y


@pytest.mark.parametrize("cal_cls", [IsotonicCalibrator, PlattCalibrator])
def test_calibrator_outputs_valid_probabilities(cal_cls):
    p_raw, y = _scores_and_labels()
    p = cal_cls().fit(p_raw, y).transform(p_raw)
    assert ((p >= 0) & (p <= 1)).all()


@pytest.mark.parametrize("cal_cls", [IsotonicCalibrator, PlattCalibrator])
def test_calibrator_is_monotone(cal_cls):
    """Ranking must survive: if raw score a >= raw score b, calibrated a >=
    calibrated b. This is what guarantees calibration cannot change PR-AUC
    ordering-based conclusions."""
    p_raw, y = _scores_and_labels()
    cal = cal_cls().fit(p_raw, y)
    grid = np.linspace(0.001, 0.999, 200)
    out = cal.transform(grid)
    assert (np.diff(out) >= -1e-12).all()


@pytest.mark.parametrize("cal_cls", [IsotonicCalibrator, PlattCalibrator])
def test_calibration_improves_brier_on_overconfident_scores(cal_cls):
    """On held-out data from the same overconfident generator, calibration must
    reduce Brier - that is its one job. (Generator is seeded; not flaky.)"""
    p_fit, y_fit = _scores_and_labels(seed=9)
    p_hold, y_hold = _scores_and_labels(seed=10)
    cal = cal_cls().fit(p_fit, y_fit)
    raw = brier_score_loss(y_hold, p_hold)
    calibrated = brier_score_loss(y_hold, cal.transform(p_hold))
    assert calibrated < raw


def test_platt_is_strictly_monotone():
    """The shipped-calibrator rule rests on this: Platt must never map two
    distinct scores to the same probability (no ties, so provably zero
    ranking damage). Isotonic deliberately has no such test - it ties."""
    p_raw, y = _scores_and_labels()
    cal = PlattCalibrator().fit(p_raw, y)
    grid = np.linspace(0.001, 0.999, 500)
    out = cal.transform(grid)
    assert (np.diff(out) > 0).all()


def test_calibration_split_is_temporal():
    """The cal slice must sit strictly after the fit slice in first-seen time,
    the same guarantee the main train/test split gives."""
    cfg = config.Settings()
    df = data.engineer_time_features(data.make_synthetic(n_rows=4000, seed=13))
    train_df, _ = data.temporal_grouped_split(df, test_size=cfg.test_size)
    fit_df, cal_df = data.temporal_grouped_split(train_df, test_size=cfg.calibration_size)
    last_fit_cohort = fit_df.groupby("entity_id")["timestamp"].min().max()
    first_cal_cohort = cal_df.groupby("entity_id")["timestamp"].min().min()
    assert last_fit_cohort <= first_cal_cohort
    assert set(fit_df["entity_id"]).isdisjoint(set(cal_df["entity_id"]))


def test_curve_from_scores_matches_hand_computation():
    """Six rows small enough to verify on paper."""
    p = np.array([0.95, 0.85, 0.55, 0.45, 0.10, 0.05])
    y = np.array([1, 0, 1, 1, 0, 1])
    (row,) = curve_from_scores(p, y, [0.8])
    # confidence = [0.95, 0.85, 0.55, 0.55, 0.90, 0.95]; covered = conf >= 0.8
    # covered: rows 0,1,4,5 -> coverage 4/6; flagged (covered & p>=0.5): rows 0,1
    assert row["coverage"] == round(4 / 6, 4)
    assert row["n_sent_to_review"] == 2
    assert row["n_flagged_fraud"] == 2
    assert row["precision_on_flagged"] == 0.5  # row0 fraud, row1 legit
    assert row["fraud_caught_auto"] == 1  # row0
    assert row["fraud_in_review_queue"] == 2  # rows 2,3 (fraud, uncovered)
    assert row["fraud_missed"] == 1  # row5: covered, predicted legit, is fraud
    assert row["recall_auto"] == 0.25  # 1 of 4 frauds


def test_expected_cost_curve_matches_hand_computation():
    curve = [
        {
            "review_threshold": 0.8,
            "n_sent_to_review": 100,
            "n_flagged_fraud": 40,
            "fraud_caught_auto": 30,
            "fraud_missed": 5,
        }
    ]
    (priced,) = expected_cost_curve(
        curve, cost_missed_fraud=200.0, cost_false_flag=10.0, cost_review=2.0
    )
    # false flags = 40 - 30 = 10; cost = 5*200 + 10*10 + 100*2 = 1300
    assert priced["n_false_flags"] == 10
    assert priced["expected_cost"] == 1300.0


def test_expected_cost_requires_explicit_costs():
    """No baked-in cost assumptions, ever: the signature must force the caller
    to supply all three."""
    with pytest.raises(TypeError):
        expected_cost_curve([])  # type: ignore[call-arg]
