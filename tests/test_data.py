import pandas as pd

from ledgersentry import data


def test_synthetic_is_deterministic():
    a = data.make_synthetic()
    b = data.make_synthetic()
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_different_seed_differs():
    a = data.make_synthetic(seed=1)
    b = data.make_synthetic(seed=2)
    assert not a["amount"].equals(b["amount"])


def test_synthetic_schema_and_size():
    df = data.make_synthetic(n_rows=5000, fraud_rate=0.01)
    for col in data.CANONICAL_COLUMNS:
        assert col in df.columns
    assert len(df) == 5000
    assert set(df["is_fraud"].unique()) <= {0, 1}
    assert df["timestamp"].is_monotonic_increasing


def test_synthetic_fraud_rate_close_to_target():
    df = data.make_synthetic(n_rows=8000, fraud_rate=0.01)
    rate = df["is_fraud"].mean()
    assert 0.005 <= rate <= 0.02


def test_split_has_no_group_leakage():
    df = data.make_synthetic(n_rows=6000, fraud_rate=0.015, seed=7)
    train, test = data.temporal_grouped_split(df, test_size=0.2)
    assert set(train["entity_id"]).isdisjoint(set(test["entity_id"]))
    assert len(train) + len(test) == len(df)
    assert len(train) > 0 and len(test) > 0


def test_split_is_approximately_temporal():
    df = data.make_synthetic(n_rows=6000, fraud_rate=0.015, seed=7)
    train, test = data.temporal_grouped_split(df, test_size=0.2)
    # every train entity's cohort (first-seen time) is no later than every test
    # entity's cohort, by construction of temporal_grouped_split.
    last_train_cohort = train.groupby("entity_id")["timestamp"].min().max()
    first_test_cohort = test.groupby("entity_id")["timestamp"].min().min()
    assert last_train_cohort <= first_test_cohort


def test_feature_columns_excludes_entity_id():
    df = data.engineer_time_features(data.make_synthetic(n_rows=500))
    numeric, categorical = data.feature_columns(df)
    assert "entity_id" not in numeric
    assert "entity_id" not in categorical
    assert "amount" in numeric
    assert "category" in categorical
