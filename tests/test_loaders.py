"""
CI coverage for all four real-dataset loaders.

Each fixture in tests/fixtures/ is a tiny CSV written to the TRUE public schema
of its dataset (real column names, realistic value shapes, fake rows), so the
loaders are exercised on every CI run without shipping or downloading any real
data. Every loader is driven through the public `load()` entry point (file
detection included), then through the full split + preprocessor path, so a
schema regression in any loader turns CI red instead of failing silently at
the first real-data run.
"""
import shutil
from pathlib import Path

import pandas as pd
import pytest

from ledgersentry import data

FIXTURES = Path(__file__).parent / "fixtures"


def _stage(tmp_path, mapping):
    """Copy fixtures into a temp data dir under the filenames _detect_real expects."""
    for fixture_name, data_name in mapping.items():
        shutil.copy(FIXTURES / fixture_name, tmp_path / data_name)
    return tmp_path


def _assert_canonical(df):
    for col in data.CANONICAL_COLUMNS:
        assert col in df.columns
    assert df["timestamp"].is_monotonic_increasing
    assert set(df["is_fraud"].unique()) <= {0, 1}
    assert pd.api.types.is_string_dtype(df["entity_id"])
    assert pd.api.types.is_float_dtype(df["amount"])


def _assert_survives_pipeline(df):
    """Split + preprocessor end to end: disjoint entities, both sides non-empty,
    and the preprocessor fits on train and transforms test without error."""
    df = data.engineer_time_features(df)
    train, test = data.temporal_grouped_split(df, test_size=0.5)
    assert len(train) > 0 and len(test) > 0
    assert set(train["entity_id"]).isdisjoint(set(test["entity_id"]))
    pre = data.build_preprocessor(train)
    x_train = pre.fit_transform(train)
    x_test = pre.transform(test)
    assert x_train.shape[0] == len(train)
    assert x_test.shape[0] == len(test)


def test_ulb_loader(tmp_path):
    _stage(tmp_path, {"ulb_creditcard.csv": "creditcard.csv"})
    df = data.load(data_dir=tmp_path)
    assert df.attrs["source"] == "ulb_creditcard"
    _assert_canonical(df)
    assert len(df) == 12
    assert int(df["is_fraud"].sum()) == 2
    # all 28 anonymized PCA components come through as f_* features
    v_cols = [c for c in df.columns if c.startswith("f_V")]
    assert len(v_cols) == 28
    # ULB has no card/customer id: every row is its own entity (documented)
    assert df["entity_id"].nunique() == len(df)
    _assert_survives_pipeline(df)


def test_sparkov_loader_concatenates_train_and_test(tmp_path):
    _stage(
        tmp_path,
        {
            "sparkov_fraudTrain.csv": "fraudTrain.csv",
            "sparkov_fraudTest.csv": "fraudTest.csv",
        },
    )
    df = data.load(data_dir=tmp_path)
    assert df.attrs["source"] == "sparkov"
    _assert_canonical(df)
    assert len(df) == 14  # 10 train rows + 4 test rows, re-split by entity later
    assert int(df["is_fraud"].sum()) == 2
    assert df["entity_id"].nunique() == 7  # cc_num is the entity key
    assert df["category"].notna().all()
    assert "f_city_pop" in df.columns
    _assert_survives_pipeline(df)


def test_ieee_cis_loader_joins_identity(tmp_path):
    _stage(
        tmp_path,
        {
            "ieee_train_transaction.csv": "train_transaction.csv",
            "ieee_train_identity.csv": "train_identity.csv",
        },
    )
    df = data.load(data_dir=tmp_path)
    assert df.attrs["source"] == "ieee_cis"
    _assert_canonical(df)
    assert len(df) == 12  # left join on TransactionID must not duplicate rows
    assert int(df["is_fraud"].sum()) == 2
    assert df["entity_id"].nunique() == 6  # card1 is the entity proxy
    assert set(df["category"].unique()) <= {"W", "H", "C"}  # ProductCD
    c_cols = [c for c in df.columns if c.startswith("f_C")]
    assert len(c_cols) == 14
    _assert_survives_pipeline(df)


def test_fdb_loader_maps_real_amount_column(tmp_path):
    _stage(tmp_path, {"fdb_train.csv": "fdb_train.csv", "fdb_test.csv": "fdb_test.csv"})
    df = data.load(data_dir=tmp_path)
    assert df.attrs["source"] == "amazon_fdb"
    _assert_canonical(df)
    assert len(df) == 14
    assert int(df["is_fraud"].sum()) == 2
    # regression test for the old bug: `amount` used to fall back to a constant
    # 0.0 because only the IEEE-CIS column name was checked. It must carry the
    # sub-dataset's real amounts (here the sparknov-style `amt` column) ...
    assert df["amount"].sum() == pytest.approx(966.78)
    assert (df["amount"] != 0.0).all()
    # ... exactly once: mapped to canonical `amount`, not duplicated as f_amt.
    assert "f_amt" not in df.columns
    assert "f_city_pop" in df.columns
    _assert_survives_pipeline(df)


def test_fdb_loader_without_amount_column_is_nan_not_zero(tmp_path):
    # FDB sub-datasets that are not transactions (fakejob, malurl, ...) have no
    # amount-like column at all. amount must be honestly missing (NaN, which the
    # model handles natively), never a fake constant.
    (tmp_path / "fdb_train.csv").write_text(
        "EVENT_ID,ENTITY_TYPE,ENTITY_ID,EVENT_TIMESTAMP,EVENT_LABEL,title,num_links\n"
        "ev_1,user,u_1,2021-01-01 10:00:00,0,offer one,3\n"
        "ev_2,user,u_2,2021-01-01 11:00:00,1,offer two,17\n"
        "ev_3,user,u_3,2021-01-01 12:00:00,0,offer three,1\n"
        "ev_4,user,u_4,2021-01-01 13:00:00,0,offer four,2\n"
    )
    df = data.load(data_dir=tmp_path)
    assert df.attrs["source"] == "amazon_fdb"
    assert df["amount"].isna().all()
    assert "f_num_links" in df.columns


def test_detection_order_prefers_sparkov_over_ulb(tmp_path):
    _stage(
        tmp_path,
        {
            "sparkov_fraudTrain.csv": "fraudTrain.csv",
            "ulb_creditcard.csv": "creditcard.csv",
        },
    )
    df = data.load(data_dir=tmp_path)
    assert df.attrs["source"] == "sparkov"


def test_empty_data_dir_falls_back_to_synthetic(tmp_path):
    df = data.load(data_dir=tmp_path)
    assert df.attrs["source"] == "synthetic"
    assert isinstance(df, pd.DataFrame)
