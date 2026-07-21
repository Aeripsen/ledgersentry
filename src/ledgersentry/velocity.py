"""
Rolling velocity/aggregate features over the transaction stream.

Counting how much activity happened in the recent past is the most standard
feature family in production card fraud (card testing, bursts of small
authorizations, an amount far above the recent norm), and this repo had none of
it: the only feature touching time was hour_of_day / day_of_week.

Two levels are computed, both strictly causal:

  stream level   over every transaction in the source, in time order
  entity level   the same windows within one entity_id (card/customer)

The entity level is the one the fraud literature means by "velocity". It is
only produced when entities actually repeat in the data. On ULB they do not:
that set publishes no card or customer id, so data.py gives every row its own
entity_id and the entity family degenerates to all-zeros. The function skips it
there rather than emitting 16 constant columns, and what ULB gets is the
stream-level family only. That is a weaker version of the standard family and
it is labeled as such wherever the numbers are reported.

Causality: every window uses `closed="left"`, so a row sees only transactions
with a STRICTLY earlier timestamp. The current row is never in its own window,
and rows sharing its timestamp are not either. That costs a little signal on
ULB, where Time is whole seconds and ties are common, but it makes the features
independent of row order within a second, which keeps them deterministic and
keeps a tie from acting as a peek sideways.

Serving note, because this is not free: these features need the recent stream to
compute, and the service in this repo is stateless. Turning them on for real
means a feature store holding trailing counts per key. That cost is why they are
opt-in rather than the default, and why the measured payoff (see
`artifacts/velocity_ulb_creditcard.json`) is worth knowing before paying it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FEATURE_PREFIX

DEFAULT_WINDOWS: tuple[str, ...] = ("1min", "5min", "1h", "24h")


def _suffix(window: str) -> str:
    return window.replace(" ", "")


def _rolling_block(
    frame: pd.DataFrame, windows: tuple[str, ...], prefix: str
) -> dict[str, np.ndarray]:
    """count / sum / mean of amount over each trailing window, plus the current
    amount as a multiple of that window's mean. `frame` must be time-sorted."""
    amount = pd.Series(frame["amount"].to_numpy(), index=frame["timestamp"])
    out: dict[str, np.ndarray] = {}
    for w in windows:
        roll = amount.rolling(w, closed="left")
        count = roll.count().to_numpy()
        total = roll.sum().to_numpy()
        count = np.nan_to_num(count, nan=0.0)
        total = np.nan_to_num(total, nan=0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
            ratio = np.where(
                (count > 0) & (mean > 0), frame["amount"].to_numpy() / mean, np.nan
            )
        s = _suffix(w)
        out[f"{prefix}count_{s}"] = count
        out[f"{prefix}amt_sum_{s}"] = total
        out[f"{prefix}amt_mean_{s}"] = mean
        out[f"{prefix}amt_ratio_{s}"] = ratio
    return out


def add_velocity_features(
    df: pd.DataFrame, windows: tuple[str, ...] = DEFAULT_WINDOWS
) -> pd.DataFrame:
    """Return `df` with the velocity columns appended, as f_* so the existing
    feature_columns()/preprocessor path picks them up unchanged. Input is
    assumed time-sorted, which every loader guarantees via data._finalize."""
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError(
            "add_velocity_features needs a time-sorted frame; data._finalize sorts "
            "every loader's output, so an unsorted frame here means something "
            "reordered it after loading"
        )
    out = df.copy()
    for name, values in _rolling_block(df, windows, f"{FEATURE_PREFIX}vel_").items():
        out[name] = values

    ts = df["timestamp"].to_numpy()
    gap = np.diff(ts).astype("timedelta64[s]").astype(float)
    out[f"{FEATURE_PREFIX}vel_seconds_since_prev"] = np.concatenate([[np.nan], gap])

    if df["entity_id"].nunique() < len(df):
        blocks = []
        for _, g in df.groupby("entity_id", sort=False):
            block = pd.DataFrame(
                _rolling_block(g, windows, f"{FEATURE_PREFIX}ent_"), index=g.index
            )
            block[f"{FEATURE_PREFIX}ent_seconds_since_prev"] = (
                g["timestamp"].diff().dt.total_seconds()
            )
            blocks.append(block)
        entity_block = pd.concat(blocks).reindex(df.index)
        for col in entity_block.columns:
            out[col] = entity_block[col].to_numpy()

    out.attrs.update(df.attrs)
    return out


def velocity_columns(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith(f"{FEATURE_PREFIX}vel_")
                  or c.startswith(f"{FEATURE_PREFIX}ent_"))
