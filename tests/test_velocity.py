import numpy as np
import pandas as pd
import pytest

from ledgersentry import data, velocity


def _frame(times, entities, amounts):
    return pd.DataFrame(
        {
            "transaction_id": [str(i) for i in range(len(times))],
            "timestamp": pd.to_datetime(times),
            "entity_id": entities,
            "amount": [float(a) for a in amounts],
            "category": None,
            "is_fraud": [0] * len(times),
        }
    )


def test_counts_and_sums_match_hand_computation():
    df = _frame(
        [
            "2026-01-01 00:00:00",
            "2026-01-01 00:00:30",
            "2026-01-01 00:02:00",
        ],
        ["e1", "e1", "e1"],
        [10, 30, 40],
    )
    out = velocity.add_velocity_features(df, windows=("1min",))
    assert list(out["f_vel_count_1min"]) == [0.0, 1.0, 0.0]
    assert list(out["f_vel_amt_sum_1min"]) == [0.0, 10.0, 0.0]
    assert out["f_vel_amt_mean_1min"].tolist()[1] == 10.0
    assert out["f_vel_amt_ratio_1min"].tolist()[1] == 3.0
    assert list(out["f_vel_seconds_since_prev"])[1:] == [30.0, 90.0]


def test_a_row_never_sees_its_own_timestamp():
    """closed='left': ties are excluded, so two transactions in the same second
    each see a count of zero rather than each other."""
    df = _frame(["2026-01-01 00:00:00"] * 2, ["e1", "e2"], [10, 20])
    out = velocity.add_velocity_features(df, windows=("1min",))
    assert list(out["f_vel_count_1min"]) == [0.0, 0.0]


def test_features_do_not_change_when_future_rows_are_appended():
    past = _frame(
        ["2026-01-01 00:00:00", "2026-01-01 00:00:10", "2026-01-01 00:00:20"],
        ["e1", "e2", "e1"],
        [10, 20, 30],
    )
    future = _frame(
        ["2026-01-01 00:00:30", "2026-01-01 00:00:40"], ["e1", "e2"], [999, 888]
    )
    both = pd.concat([past, future], ignore_index=True)

    a = velocity.add_velocity_features(past)
    b = velocity.add_velocity_features(both).iloc[: len(past)]
    for col in velocity.velocity_columns(a):
        np.testing.assert_allclose(
            a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float),
            err_msg=f"{col} changed when later transactions were added",
        )


def test_entity_block_is_skipped_when_every_row_is_its_own_entity():
    """The ULB shape: no card id, so data.py gives each row a unique entity_id
    and the per-entity family would be all zeros."""
    df = _frame(
        ["2026-01-01 00:00:00", "2026-01-01 00:00:10"], ["0", "1"], [10, 20]
    )
    out = velocity.add_velocity_features(df)
    assert not [c for c in out.columns if c.startswith("f_ent_")]
    assert [c for c in out.columns if c.startswith("f_vel_")]


def test_entity_windows_only_count_the_same_entity():
    df = _frame(
        [
            "2026-01-01 00:00:00",
            "2026-01-01 00:00:10",
            "2026-01-01 00:00:20",
        ],
        ["e1", "e2", "e1"],
        [10, 20, 30],
    )
    out = velocity.add_velocity_features(df, windows=("1min",))
    assert list(out["f_vel_count_1min"]) == [0.0, 1.0, 2.0]
    assert list(out["f_ent_count_1min"]) == [0.0, 0.0, 1.0]
    assert list(out["f_ent_amt_sum_1min"]) == [0.0, 0.0, 10.0]


def test_columns_are_picked_up_as_model_features():
    df = data.engineer_time_features(data.make_synthetic(n_rows=500))
    out = velocity.add_velocity_features(df)
    numeric, _ = data.feature_columns(out)
    for col in velocity.velocity_columns(out):
        assert col in numeric


def test_unsorted_input_is_rejected():
    df = _frame(
        ["2026-01-01 00:01:00", "2026-01-01 00:00:00"], ["e1", "e1"], [10, 20]
    )
    with pytest.raises(ValueError, match="time-sorted"):
        velocity.add_velocity_features(df)


def test_is_deterministic():
    df = data.make_synthetic(n_rows=800, seed=3)
    pd.testing.assert_frame_equal(
        velocity.add_velocity_features(df), velocity.add_velocity_features(df)
    )
