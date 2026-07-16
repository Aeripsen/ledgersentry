"""
The compiled scoring path is only allowed to exist because it changes nothing:
these tests pin its transform output and decisions to the pandas/ColumnTransformer
reference path, exactly, on every kind of input the service can see. The last
test is the latency regression guard: if someone reintroduces per-row pandas
work into the compiled path, CI goes red before the README's numbers go stale.
"""
import time

import numpy as np
import pytest

from ledgersentry import data
from ledgersentry.model import FraudDetector
from ledgersentry.scoring import CompiledScorer, PandasScorer


@pytest.fixture(scope="module")
def fitted():
    """Preprocessor + model fitted on the synthetic pipeline (has BOTH a
    categorical column and numeric f_* columns, so the one-hot block and the
    passthrough block are both exercised)."""
    raw = data.make_synthetic(n_rows=3000, fraud_rate=0.02, seed=3)
    df = data.engineer_time_features(raw)
    train_df, test_df = data.temporal_grouped_split(df, test_size=0.25)
    pre = data.build_preprocessor(train_df)
    X_train = pre.fit_transform(train_df)
    model = FraudDetector(max_iter=50).fit(X_train, train_df["is_fraud"].to_numpy())
    return pre, model, test_df


def test_transform_frame_matches_column_transformer_exactly(fitted):
    pre, model, test_df = fitted
    compiled = CompiledScorer(pre, model)
    expected = pre.transform(test_df)
    got = compiled.transform_frame(test_df)
    assert got.shape == expected.shape
    assert np.array_equal(got, np.asarray(expected, dtype=np.float64), equal_nan=True)


def test_transform_one_matches_reference_on_full_rows(fitted):
    pre, model, test_df = fitted
    compiled = CompiledScorer(pre, model)
    reference = PandasScorer(pre, model)
    cols = compiled.numeric_cols + compiled.categorical_cols
    for _, row in test_df.head(25)[cols].iterrows():
        features = row.to_dict()
        got = compiled.transform_one(features)
        expected = pre.transform(reference._frame_one(features))
        assert np.array_equal(got, np.asarray(expected, dtype=np.float64), equal_nan=True)


def test_transform_one_missing_numeric_is_nan(fitted):
    pre, model, _ = fitted
    compiled = CompiledScorer(pre, model)
    vec = compiled.transform_one({"amount": 50.0, "category": "gas"})
    k = compiled.numeric_cols.index("f_entity_daily_tx_count")
    assert np.isnan(vec[0, compiled._num_offset + k])
    a = compiled.numeric_cols.index("amount")
    assert vec[0, compiled._num_offset + a] == 50.0


def test_unknown_category_one_hots_to_all_zeros(fitted):
    """handle_unknown='ignore' parity: an unseen category must produce an
    all-zero one-hot block in both paths, not an error."""
    pre, model, _ = fitted
    compiled = CompiledScorer(pre, model)
    reference = PandasScorer(pre, model)
    features = {"amount": 10.0, "hour_of_day": 3, "day_of_week": 2,
                "f_entity_daily_tx_count": 1, "category": "never-seen-this"}
    got = compiled.transform_one(features)
    expected = pre.transform(reference._frame_one(features))
    assert np.array_equal(got, np.asarray(expected, dtype=np.float64), equal_nan=True)
    onehot_width = compiled._num_offset
    assert onehot_width > 0
    assert (got[0, :onehot_width] == 0.0).all()


@pytest.mark.parametrize("threshold", [0.0, 0.7, 0.95])
def test_decisions_identical_across_paths(fitted, threshold):
    pre, model, test_df = fitted
    compiled = CompiledScorer(pre, model)
    reference = PandasScorer(pre, model)
    got = compiled.score_frame(test_df, review_threshold=threshold)
    expected = reference.score_frame(test_df, review_threshold=threshold)
    assert np.array_equal(got.decisions, expected.decisions)
    assert np.allclose(got.p_fraud, expected.p_fraud)
    assert np.allclose(got.confidence, expected.confidence)


def test_score_one_matches_score_frame(fitted):
    pre, model, test_df = fitted
    compiled = CompiledScorer(pre, model)
    cols = compiled.numeric_cols + compiled.categorical_cols
    sample = test_df.head(10)
    frame_result = compiled.score_frame(sample)
    for i, (_, row) in enumerate(sample[cols].iterrows()):
        one = compiled.score_one(row.to_dict())
        assert one.decision == str(frame_result.decisions[i])
        assert one.p_fraud == pytest.approx(float(frame_result.p_fraud[i]))


def test_missing_frame_column_fills_nan(fitted):
    pre, model, test_df = fitted
    compiled = CompiledScorer(pre, model)
    dropped = test_df.drop(columns=["f_entity_daily_tx_count"])
    mat = compiled.transform_frame(dropped)
    k = compiled.numeric_cols.index("f_entity_daily_tx_count")
    assert np.isnan(mat[:, compiled._num_offset + k]).all()


def test_compiled_single_row_latency_guard(fitted):
    """Regression guard, deliberately loose: compiled single-row p95 must stay
    under 5 ms. Measured ~0.3-1 ms on a 2023 laptop; the pre-optimization
    pandas path measured ~8-12 ms, so a regression to per-row DataFrame work
    trips this even on a slow CI runner."""
    pre, model, test_df = fitted
    compiled = CompiledScorer(pre, model)
    cols = compiled.numeric_cols + compiled.categorical_cols
    rows = test_df.head(200)[cols].to_dict("records")
    for row in rows[:20]:  # warmup
        compiled.score_one(row)
    latencies = []
    for row in rows:
        t0 = time.perf_counter()
        compiled.score_one(row)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    p95 = float(np.percentile(latencies, 95))
    assert p95 < 5.0, f"compiled single-row p95 regressed to {p95:.2f} ms"


def test_pandas_scorer_types():
    """PandasScorer must build its 1-row frame with NaN (not 0) for missing
    numerics and 'unknown' for missing category - the service's honesty rule."""
    raw = data.make_synthetic(n_rows=400, seed=5)
    df = data.engineer_time_features(raw)
    pre = data.build_preprocessor(df)
    pre.fit(df)
    model = FraudDetector(max_iter=10).fit(pre.transform(df), df["is_fraud"].to_numpy())
    scorer = PandasScorer(pre, model)
    frame = scorer._frame_one({"amount": 12.5})
    assert np.isnan(frame["f_entity_daily_tx_count"].iloc[0])
    assert frame["category"].iloc[0] == "unknown"
