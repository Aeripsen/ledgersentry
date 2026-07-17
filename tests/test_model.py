import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from ledgersentry import data
from ledgersentry.model import (
    REVIEW,
    FraudDetector,
    curve_from_scores,
    decoupled_curve_from_scores,
)


def _fit_on_synthetic(n_rows=4000, fraud_rate=0.03, seed=1):
    """Small, fast fixture: synthetic loader -> time features -> leakage-safe
    split -> preprocessor fit on TRAIN only -> model fit on TRAIN only."""
    raw = data.make_synthetic(n_rows=n_rows, fraud_rate=fraud_rate, seed=seed)
    df = data.engineer_time_features(raw)
    train_df, test_df = data.temporal_grouped_split(df, test_size=0.25)
    pre = data.build_preprocessor(train_df)
    X_train = pre.fit_transform(train_df)
    X_test = pre.transform(test_df)
    model = FraudDetector().fit(X_train, train_df["is_fraud"].to_numpy())
    return model, X_test, test_df["is_fraud"].to_numpy()


def test_model_predicts_valid_probabilities():
    model, X_test, y_test = _fit_on_synthetic()
    p = model.predict_proba_fraud(X_test)
    assert len(p) == len(y_test)
    assert ((p >= 0) & (p <= 1)).all()


def test_pr_auc_in_valid_range():
    model, X_test, y_test = _fit_on_synthetic()
    p = model.predict_proba_fraud(X_test)
    assert y_test.sum() > 0, "test fixture must contain at least one fraud row"
    pr_auc = average_precision_score(y_test, p)
    assert 0.0 < pr_auc < 1.0


def test_pr_auc_beats_random_baseline():
    """The model should do meaningfully better than a no-skill scorer (whose
    PR-AUC equals the positive rate) - otherwise it learned nothing."""
    model, X_test, y_test = _fit_on_synthetic()
    p = model.predict_proba_fraud(X_test)
    pr_auc = average_precision_score(y_test, p)
    random_baseline = y_test.mean()
    assert pr_auc > random_baseline


def test_reject_knob_abstains_on_low_confidence():
    model, X_test, y_test = _fit_on_synthetic()
    decision, p_fraud, confidence = model.decide(X_test, review_threshold=0.999)
    assert (decision == REVIEW).any(), "a near-1.0 review threshold should send some rows to review"


def test_reject_knob_answers_everything_at_zero_threshold():
    model, X_test, y_test = _fit_on_synthetic()
    decision, p_fraud, confidence = model.decide(X_test, review_threshold=0.0)
    assert not (decision == REVIEW).any()


def test_coverage_decreases_as_threshold_rises():
    model, X_test, y_test = _fit_on_synthetic()
    curve = model.coverage_precision_curve(X_test, y_test, [0.5, 0.7, 0.9, 0.99])
    coverage = [r["coverage"] for r in curve]
    assert coverage == sorted(coverage, reverse=True)


def test_coverage_precision_curve_row_shape():
    model, X_test, y_test = _fit_on_synthetic()
    curve = model.coverage_precision_curve(X_test, y_test, [0.5, 0.9])
    total_fraud = int((y_test == 1).sum())
    for row in curve:
        assert 0.0 <= row["coverage"] <= 1.0
        assert row["precision_on_flagged"] is None or 0.0 <= row["precision_on_flagged"] <= 1.0
        assert isinstance(row["n_flagged_fraud"], int)
        # recall side: the three fraud_* buckets must partition every true fraud
        assert (
            row["fraud_caught_auto"] + row["fraud_in_review_queue"] + row["fraud_missed"]
            == total_fraud
        )
        assert 0.0 <= row["recall_auto"] <= 1.0


def test_full_coverage_recall_matches_caught_over_total():
    """At threshold 0.5 nothing is sent to review, so recall_auto is exactly the
    frauds auto-caught over all test frauds - the headline number the README quotes."""
    model, X_test, y_test = _fit_on_synthetic()
    row = model.coverage_precision_curve(X_test, y_test, [0.5])[0]
    total_fraud = int((y_test == 1).sum())
    assert row["fraud_in_review_queue"] == 0  # full coverage -> empty review queue
    assert row["recall_auto"] == round(row["fraud_caught_auto"] / total_fraud, 4)


def test_sample_weight_favors_minority_class():
    """Balanced weighting should give the ~1% fraud class much higher per-row
    weight than the majority class - the mechanism the model relies on instead
    of resampling."""
    y = np.array([0] * 990 + [1] * 10)
    counts = np.bincount(y)
    weight_per_class = counts.sum() / (len(counts) * np.maximum(counts, 1))
    sample_weight = weight_per_class[y]
    assert sample_weight[y == 1].mean() > sample_weight[y == 0].mean()


def test_decoupled_curve_generalizes_the_symmetric_one():
    """The load-bearing claim: the decoupled curve is a strict SUPERSET of the
    symmetric one, not a reimplementation that drifts from it.

    For any t > 0.5, curve_from_scores(t) already means "flag if p >= t, clear
    if p <= 1-t", so (flag_at=t, clear_at=1-t) must reproduce it row for row.
    t=0.5 is excluded on purpose: there the two cuts would collide at exactly
    0.5 and the lanes would overlap, which is why the symmetric knob's 0.5 row
    is a special case and why flag_at > clear_at is enforced.
    """
    rng = np.random.default_rng(11)
    p = rng.random(2000)
    y = (rng.random(2000) < 0.05).astype(int)
    shared = ("coverage", "n_sent_to_review", "n_flagged_fraud",
              "precision_on_flagged", "fraud_caught_auto",
              "fraud_in_review_queue", "fraud_missed", "recall_auto")
    for t in (0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0):
        (sym,) = curve_from_scores(p, y, [t])
        (dec,) = decoupled_curve_from_scores(p, y, [(t, 1 - t)])
        for key in shared:
            assert sym[key] == dec[key], f"t={t} diverged on {key}"


def test_decoupled_curve_expresses_what_the_symmetric_knob_cannot():
    """The reason this exists: flag high AND clear low at the same time. The
    symmetric knob forces clear_at = 1 - flag_at, so asking to flag at 0.9
    demands clearing at 0.1; here the two are independent."""
    rng = np.random.default_rng(12)
    p = rng.random(4000)
    y = (rng.random(4000) < 0.05).astype(int)
    (row,) = decoupled_curve_from_scores(p, y, [(0.9, 0.02)])
    assert row["flag_at"] == 0.9
    assert row["clear_at"] == 0.02
    # everything flagged is >= 0.9, everything cleared is <= 0.02, and the band
    # between goes to review - a point (0.9, 0.1) is the only thing the
    # symmetric knob could have offered
    assert row["n_flagged_fraud"] == int((p >= 0.9).sum())
    n_reviewed = int(((p > 0.02) & (p < 0.9)).sum())
    assert row["n_sent_to_review"] == n_reviewed


def test_decoupled_curve_partitions_every_fraud():
    """Same invariant the symmetric table guarantees: caught + queued + missed
    must account for every true fraud at every operating point, so the table
    cannot be read selectively."""
    rng = np.random.default_rng(13)
    p = rng.random(3000)
    y = (rng.random(3000) < 0.08).astype(int)
    total = int((y == 1).sum())
    points = [(0.9, 0.02), (0.8, 0.01), (0.55, 0.45), (1.0, 0.0)]
    for row in decoupled_curve_from_scores(p, y, points):
        assert (
            row["fraud_caught_auto"]
            + row["fraud_in_review_queue"]
            + row["fraud_missed"]
        ) == total


def test_decoupled_curve_rejects_overlapping_lanes():
    """A row that is both auto-flagged and auto-cleared is a contradiction, not
    an operating point. It must be a loud error, never a silent precedence."""
    p = np.array([0.9, 0.1])
    y = np.array([1, 0])
    for bad in [(0.5, 0.5), (0.3, 0.8)]:
        with pytest.raises(ValueError, match="strictly greater"):
            decoupled_curve_from_scores(p, y, [bad])


def test_decoupled_curve_matches_hand_computation():
    """Six rows small enough to verify on paper, the same way the symmetric
    curve is pinned."""
    p = np.array([0.95, 0.85, 0.55, 0.45, 0.10, 0.05])
    y = np.array([1, 0, 1, 1, 0, 1])
    (row,) = decoupled_curve_from_scores(p, y, [(0.9, 0.06)])
    # flagged (p >= 0.9): row0 -> 1 row, truly fraud -> precision 1.0
    # cleared (p <= 0.06): row5 -> 1 row, and it IS fraud -> a real miss
    # review (0.06 < p < 0.9): rows 1,2,3,4 -> 4 rows, frauds among them: 2,3
    assert row["coverage"] == round(2 / 6, 4)
    assert row["n_sent_to_review"] == 4
    assert row["n_flagged_fraud"] == 1
    assert row["precision_on_flagged"] == 1.0
    assert row["fraud_caught_auto"] == 1
    assert row["fraud_in_review_queue"] == 2
    assert row["fraud_missed"] == 1
    assert row["recall_auto"] == 0.25  # 1 of 4 frauds
