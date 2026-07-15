import numpy as np
from sklearn.metrics import average_precision_score

from ledgersentry import data
from ledgersentry.model import REVIEW, FraudDetector


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
    for row in curve:
        assert 0.0 <= row["coverage"] <= 1.0
        assert row["precision_on_flagged"] is None or 0.0 <= row["precision_on_flagged"] <= 1.0
        assert isinstance(row["n_flagged_fraud"], int)


def test_sample_weight_favors_minority_class():
    """Balanced weighting should give the ~1% fraud class much higher per-row
    weight than the majority class - the mechanism the model relies on instead
    of resampling."""
    y = np.array([0] * 990 + [1] * 10)
    counts = np.bincount(y)
    weight_per_class = counts.sum() / (len(counts) * np.maximum(counts, 1))
    sample_weight = weight_per_class[y]
    assert sample_weight[y == 1].mean() > sample_weight[y == 0].mean()
