import json

import numpy as np
import pandas as pd
import pytest

from ledgersentry import data
from ledgersentry.bootstrap import wilson_interval
from ledgersentry.model import curve_from_scores
from ledgersentry.rolling import (
    build_folds,
    cut_timestamp,
    evaluate,
    expected_calibration_error,
    split_before,
    threshold_metrics,
)

MODEL_KW = {"random_state": 42, "max_iter": 30, "learning_rate": 0.1, "model": "hist_gbdt"}


def _frame(n_rows=6000, fraud_rate=0.04, seed=3):
    return data.engineer_time_features(
        data.make_synthetic(n_rows=n_rows, fraud_rate=fraud_rate, seed=seed)
    )


def _row_per_entity(timestamps, labels):
    """ULB-shaped frame: every row its own entity."""
    n = len(timestamps)
    return pd.DataFrame({
        "transaction_id": [str(i) for i in range(n)],
        "timestamp": pd.to_datetime(timestamps),
        "entity_id": [str(i) for i in range(n)],
        "amount": np.ones(n),
        "category": None,
        "is_fraud": labels,
    })


# --------------------------------------------------------------------------- #
# folds
# --------------------------------------------------------------------------- #

def test_folds_expand_never_straddle_and_tile_the_tail():
    df = _frame()
    folds = build_folds(df, (0.2, 0.4, 0.6, 0.8), min_test_fraud=30)
    assert len(folds) == 4
    first = df.groupby("entity_id")["timestamp"].transform("min")
    covered = np.zeros(len(df), dtype=bool)
    for prev, fold in zip([None, *folds], folds, strict=False):
        train_ids = set(df.loc[fold.train_mask, "entity_id"])
        test_ids = set(df.loc[fold.test_mask, "entity_id"])
        assert not train_ids & test_ids, "an entity straddles the cut"
        assert (first[fold.train_mask] < fold.cut).all()
        assert (first[fold.test_mask] >= fold.cut).all()
        assert int(df.loc[fold.test_mask, "is_fraud"].sum()) >= 30
        assert not (covered & fold.test_mask).any(), "test windows overlap"
        covered |= fold.test_mask
        if prev is not None:
            # expanding: the previous train window plus the previous test window
            assert np.array_equal(fold.train_mask, prev.train_mask | prev.test_mask)
    assert np.array_equal(covered, ~folds[0].train_mask)


def test_last_fold_equals_the_headline_split_when_no_second_is_shared():
    df = _frame()
    train_df, test_df = data.temporal_grouped_split(df, test_size=0.2)
    cut = cut_timestamp(df, 0.8)
    mine_train, mine_test = split_before(df, cut)
    assert set(mine_train["transaction_id"]) == set(train_df["transaction_id"])
    assert set(mine_test["transaction_id"]) == set(test_df["transaction_id"])


def test_a_shared_second_goes_wholly_to_the_later_side():
    # 10 rows, the 8th and 9th share a second; an 0.8 budget would split them
    ts = [f"2013-01-01 00:00:{s:02d}" for s in (0, 1, 2, 3, 4, 5, 6, 7, 7, 9)]
    df = _row_per_entity(ts, [0] * 10)
    cut = cut_timestamp(df, 0.8)
    train, test = split_before(df, cut)
    assert cut == pd.Timestamp("2013-01-01 00:00:07")
    assert len(train) == 7 and len(test) == 3
    assert train["timestamp"].max() < test["timestamp"].min()


def test_short_windows_merge_forward_and_the_last_merges_back():
    ts = pd.date_range("2013-01-01", periods=100, freq="s")
    y = np.zeros(100, dtype=int)
    y[[1, 2]] = 1          # fold-1 train
    y[[25, 30]] = 1        # window 1 (rows 20-39): 2 frauds, short
    y[[45, 50, 55]] = 1    # window 2 (rows 40-59): 3 frauds
    y[[65, 70, 75]] = 1    # window 3 (rows 60-79): 3 frauds
    y[[85]] = 1            # window 4 (rows 80-99): 1 fraud, short
    folds = build_folds(_row_per_entity(ts, y), (0.2, 0.4, 0.6, 0.8), min_test_fraud=3)
    sizes = [(int(f.test_mask.sum()), f.merged_windows) for f in folds]
    # window 1 merges into window 2; window 4 merges back into window 3
    assert sizes == [(40, 2), (40, 2)]
    assert [f.cut for f in folds] == [ts[20], ts[60]]


def test_too_few_frauds_after_the_first_cut_is_an_error():
    ts = pd.date_range("2013-01-01", periods=50, freq="s")
    y = np.zeros(50, dtype=int)
    y[[1, 40]] = 1
    with pytest.raises(ValueError, match="cannot reach"):
        build_folds(_row_per_entity(ts, y), (0.2, 0.6), min_test_fraud=5)


def test_fractions_must_increase():
    with pytest.raises(ValueError):
        build_folds(_frame(), (0.6, 0.4))


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def test_ece_by_hand():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.3, 0.6, 0.8])
    # two bins: {0.1, 0.3} vs 0 frauds -> |0.4 - 0|; {0.6, 0.8} vs 2 -> |1.4 - 2|
    assert expected_calibration_error(y, p, n_bins=2) == pytest.approx((0.4 + 0.6) / 4)
    assert expected_calibration_error(y, y.astype(float), n_bins=2) == 0.0


def test_threshold_metrics_match_the_curve_and_wilson():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.02).astype(int)
    p = np.clip(0.6 * y + rng.random(5000) * 0.5, 0, 1)
    rows = threshold_metrics(y, p, (0.5, 0.95))
    curve = curve_from_scores(p, y, (0.5, 0.95))
    for row, band in zip(rows, curve, strict=True):
        assert row["frauds_flagged"] == band["fraud_caught_auto"]
        assert row["recall"]["value"] == band["recall_auto"]
        assert row["review_band"]["frauds_cleared_unseen"] == band["fraud_missed"]
        tp, fp = row["frauds_flagged"], row["false_alerts"]
        lo, hi = wilson_interval(tp, int(y.sum()))
        assert row["recall"]["ci_lower"] == round(lo, 4)
        assert row["recall"]["ci_upper"] == round(hi, 4)
        assert row["false_alerts_per_10k"]["value"] == round(fp / len(y) * 1e4, 2)


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def report():
    return evaluate(_frame(), model_kw=MODEL_KW, n_resamples=40, seed=7)


def test_report_shape_and_calibration_slice_precedes_the_test_window(report):
    assert len(report["folds"]) == 4
    for f in report["folds"]:
        assert f["entities_straddling_cut"] == 0
        assert f["pr_auc"]["ci_lower"] <= f["pr_auc"]["value"] <= f["pr_auc"]["ci_upper"]
        assert f["pr_auc"]["value"] > f["no_skill_baseline"]
        cal = f["calibration"]
        assert cal["calibration_slice"]["n_fraud"] > 0
        assert (cal["fit_slice"]["n_rows"] + cal["calibration_slice"]["n_rows"]
                == f["train"]["n_rows"])
        assert cal["calibration_slice"]["first_seen_from"]["timestamp"] < (
            f["test"]["first_seen_from"]["timestamp"])
        assert [t["threshold"] for t in f["thresholds"]] == [0.5, 0.95]
    pooled = report["pooled"]
    assert pooled["n_fraud"] == sum(f["test"]["n_fraud"] for f in report["folds"])
    assert pooled["thresholds"][0]["frauds_flagged"] == sum(
        f["thresholds"][0]["frauds_flagged"] for f in report["folds"])


def test_report_is_deterministic(report):
    again = evaluate(_frame(), model_kw=MODEL_KW, n_resamples=40, seed=7)
    assert json.dumps(again) == json.dumps(report)
