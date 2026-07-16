"""
Dataset-agnostic transaction loader for LedgerSentry.

Canonical schema (after loading, before feature engineering):
    transaction_id : str
    timestamp      : datetime64[ns]
    entity_id      : str    (card/customer/account id - the GROUPED-split key,
                              deliberately never used as a model feature, so the
                              model has to learn from behavior, not memorize an id)
    amount         : float
    category       : str or None (merchant/transaction category, if the source has one)
    is_fraud       : int (0/1)
    f_*            : any extra numeric feature columns the source provides, kept
                      as-is (e.g. the anonymized V1..V28 PCA components on the ULB
                      set, or the C1..C14 velocity counts on IEEE-CIS).

Real sources, loaded automatically when their files are dropped in `data_dir`.
None of them ship in this repo or are downloaded by this code - IEEE-CIS and the
Amazon FDB sources are Kaggle/auth-gated, and even the open ones are tens to
hundreds of MB, so fetching is a manual, documented step (see README "Data").

    ULB "Credit Card Fraud Detection"                  data/creditcard.csv
      Time, V1..V28, Amount, Class
      https://www.kaggle.com/mlg-ulb/creditcardfraud

    Sparkov "Credit Card Transactions Fraud Detection"  data/fraudTrain.csv [+ fraudTest.csv]
    (kartik2112 on Kaggle, built with the Sparkov simulator - has real timestamps,
    which is why FINTECH_PLAN.md picks it for the streaming/real-time story)
      trans_date_trans_time, cc_num, merchant, category, amt, ..., is_fraud
      https://www.kaggle.com/datasets/kartik2112/fraud-detection

    IEEE-CIS Fraud Detection (Vesta)     data/train_transaction.csv [+ train_identity.csv]
      TransactionID, isFraud, TransactionDT, TransactionAmt, card1.., ProductCD, ...
      https://www.kaggle.com/c/ieee-fraud-detection

    Amazon `fraud-dataset-benchmark`                    data/fdb_train.csv [+ fdb_test.csv]
    The FDB python package has no file-export of its own; run it yourself and drop
    `obj.train.to_csv("data/fdb_train.csv")` (and optionally `obj.test...`). Verified
    standardized columns (github.com/amazon-science/fraud-dataset-benchmark, checked
    2026-07-14): EVENT_LABEL, EVENT_TIMESTAMP, ENTITY_ID, ENTITY_TYPE, EVENT_ID.
      https://github.com/amazon-science/fraud-dataset-benchmark

When none of these files are present, `load()` deterministically generates a
small synthetic imbalanced transaction set so training, tests, and CI are green
with zero network access and zero gated downloads. Synthetic results are always
labeled `is_synthetic: true` wherever they're reported - see docs/model_card.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

CANONICAL_COLUMNS = ["transaction_id", "timestamp", "entity_id", "amount", "category", "is_fraud"]
FEATURE_PREFIX = "f_"


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #

def _to_binary_label(s: pd.Series) -> pd.Series:
    """Normalize a fraud-label column to 0/1 ints regardless of source encoding
    (0/1, True/False, or string labels like 'fraud'/'legit')."""
    if s.dtype == bool:
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return (s != 0).astype(int)
    return s.astype(str).str.strip().str.lower().isin({"1", "true", "fraud", "yes"}).astype(int)


def _finalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Coerce dtypes, fill in the category column consistently, and sort by time.
    Every loader (real or synthetic) ends by calling this, so the rest of the
    pipeline can rely on one clean, consistent canonical frame."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["is_fraud"] = _to_binary_label(df["is_fraud"])
    df["amount"] = df["amount"].astype(float)
    df["entity_id"] = df["entity_id"].astype(str)
    df["transaction_id"] = df["transaction_id"].astype(str)
    if "category" in df.columns and df["category"].notna().any():
        df["category"] = df["category"].fillna("unknown").astype(str)
    else:
        df["category"] = None
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.attrs["source"] = source
    return df


# --------------------------------------------------------------------------- #
# real-dataset loaders (exercised once a user drops the real file in `data/`;
# none of these are downloaded or committed by this repo)
# --------------------------------------------------------------------------- #

def _load_ulb(path: Path) -> pd.DataFrame:
    """ULB European credit-card set: Time (seconds since the first transaction),
    V1..V28 (already PCA-anonymized), Amount, Class. There is no customer/card id
    on this dataset, so each row is its own entity: the grouped split degrades to
    a plain temporal split for this source only (documented, not hidden)."""
    raw = pd.read_csv(path)
    v_cols = [c for c in raw.columns if c.startswith("V")]
    out = pd.DataFrame(
        {
            "transaction_id": raw.index.astype(str),
            "timestamp": pd.to_datetime(raw["Time"], unit="s", origin="2013-01-01"),
            "entity_id": raw.index.astype(str),
            "amount": raw["Amount"],
            "category": None,
            "is_fraud": raw["Class"],
        }
    )
    for c in v_cols:
        out[f"{FEATURE_PREFIX}{c}"] = raw[c]
    return _finalize(out, "ulb_creditcard")


def _load_sparkov(train_path: Path, test_path: Path | None) -> pd.DataFrame:
    """Sparkov-generated transactions (kartik2112 Kaggle set): trans_date_trans_time
    is a real calendar timestamp, cc_num is the card/entity id, amt/category/is_fraud
    are the label columns. Concatenates fraudTrain + fraudTest (if present) into one
    frame - this loader's own temporal_grouped_split does the leakage-safe split."""
    frames = [pd.read_csv(train_path)]
    if test_path and test_path.exists():
        frames.append(pd.read_csv(test_path))
    raw = pd.concat(frames, ignore_index=True)
    tx_id = raw["trans_num"] if "trans_num" in raw.columns else raw.index.astype(str)
    out = pd.DataFrame(
        {
            "transaction_id": tx_id,
            "timestamp": pd.to_datetime(raw["trans_date_trans_time"]),
            "entity_id": raw["cc_num"].astype(str),
            "amount": raw["amt"],
            "category": raw["category"] if "category" in raw.columns else None,
            "is_fraud": raw["is_fraud"],
        }
    )
    if "city_pop" in raw.columns:
        out[f"{FEATURE_PREFIX}city_pop"] = raw["city_pop"]
    return _finalize(out, "sparkov")


def _load_ieee_cis(tx_path: Path, identity_path: Path | None) -> pd.DataFrame:
    """IEEE-CIS (Vesta) train_transaction.csv, optionally joined to train_identity.csv
    on TransactionID. TransactionDT is seconds from an undocumented reference point
    (a known quirk of this dataset, not a real calendar date) - fine for a relative
    temporal split, not for absolute dates, so we anchor it at an arbitrary origin."""
    raw = pd.read_csv(tx_path)
    if identity_path and identity_path.exists():
        raw = raw.merge(pd.read_csv(identity_path), on="TransactionID", how="left")
    out = pd.DataFrame(
        {
            "transaction_id": raw["TransactionID"].astype(str),
            "timestamp": pd.to_datetime(raw["TransactionDT"], unit="s", origin="2017-11-01"),
            "entity_id": raw["card1"].astype(str),
            "amount": raw["TransactionAmt"],
            "category": raw["ProductCD"] if "ProductCD" in raw.columns else None,
            "is_fraud": raw["isFraud"],
        }
    )
    for i in range(1, 15):
        c = f"C{i}"
        if c in raw.columns:
            out[f"{FEATURE_PREFIX}{c}"] = raw[c]
    return _finalize(out, "ieee_cis")


# FDB standardizes the label/entity/time columns but NOT an amount column - each
# sub-dataset keeps its own name (TransactionAmt on the ieeecis export, amt on
# sparknov, disbursed_amount on vehicleloan, and the text sets have none at all).
_FDB_AMOUNT_CANDIDATES = {
    "transactionamt", "amt", "amount", "transaction_amount", "transactionamount",
    "disbursed_amount",
}


def _load_fdb(train_path: Path, test_path: Path | None) -> pd.DataFrame:
    """Amazon `fraud-dataset-benchmark` standardized export. The FDB package itself
    has no CSV-export mechanism; this reads whatever you saved from its Python API
    (`obj.train.to_csv(...)`). Verified standardized columns: EVENT_LABEL,
    EVENT_TIMESTAMP, ENTITY_ID (github.com/amazon-science/fraud-dataset-benchmark)."""
    frames = [pd.read_csv(train_path)]
    if test_path and test_path.exists():
        frames.append(pd.read_csv(test_path))
    raw = pd.concat(frames, ignore_index=True)
    reserved = {
        "EVENT_LABEL", "EVENT_TIMESTAMP", "ENTITY_ID", "ENTITY_TYPE", "LABEL_TIMESTAMP", "EVENT_ID",
    }
    tx_id = raw["EVENT_ID"] if "EVENT_ID" in raw.columns else raw.index.astype(str)
    # Map the sub-dataset's own amount column (case-insensitive match against the
    # known FDB names above). When a sub-dataset has no amount-like column at all,
    # amount is NaN - honestly missing, natively handled by
    # HistGradientBoostingClassifier - never a fake constant 0.0.
    amount_col = next((c for c in raw.columns if c.lower() in _FDB_AMOUNT_CANDIDATES), None)
    out = pd.DataFrame(
        {
            "transaction_id": tx_id,
            "timestamp": pd.to_datetime(raw["EVENT_TIMESTAMP"]),
            "entity_id": raw["ENTITY_ID"].astype(str),
            "amount": raw[amount_col] if amount_col is not None else float("nan"),
            "category": None,
            "is_fraud": raw["EVENT_LABEL"],
        }
    )
    for c in raw.columns:
        if c == amount_col:
            continue  # already mapped to the canonical `amount`, don't duplicate as f_*
        if c not in reserved and pd.api.types.is_numeric_dtype(raw[c]):
            out[f"{FEATURE_PREFIX}{c}"] = raw[c]
    return _finalize(out, "amazon_fdb")


def _detect_real(data_dir: Path) -> pd.DataFrame | None:
    """First real-dataset file found in `data_dir`, checked in the order named in
    FINTECH_PLAN.md (Sparkov = streaming demo, IEEE-CIS = headline benchmark,
    Amazon FDB = comparability, ULB = classic leakage-safe baseline). Returns
    None (caller falls back to synthetic) if no real files are present."""
    sparkov_train = data_dir / "fraudTrain.csv"
    if sparkov_train.exists():
        return _load_sparkov(sparkov_train, data_dir / "fraudTest.csv")

    ieee_tx = data_dir / "train_transaction.csv"
    if ieee_tx.exists():
        return _load_ieee_cis(ieee_tx, data_dir / "train_identity.csv")

    fdb_train = data_dir / "fdb_train.csv"
    if fdb_train.exists():
        return _load_fdb(fdb_train, data_dir / "fdb_test.csv")

    ulb = data_dir / "creditcard.csv"
    if ulb.exists():
        return _load_ulb(ulb)

    return None


# --------------------------------------------------------------------------- #
# synthetic fallback - deterministic, so tests/CI/scripts/train.py are green
# with no network access and no gated download
# --------------------------------------------------------------------------- #

_CATEGORIES = np.array(
    [
        "groceries", "gas", "restaurant", "electronics",
        "travel", "online", "entertainment", "utilities",
    ]
)
_CATEGORY_P_LEGIT = np.array([0.22, 0.16, 0.16, 0.10, 0.08, 0.14, 0.08, 0.06])
_CATEGORY_P_FRAUD = np.array([0.05, 0.05, 0.05, 0.22, 0.20, 0.30, 0.08, 0.05])


def make_synthetic(
    n_rows: int = 8000,
    fraud_rate: float = 0.01,
    n_entities: int = 400,
    days: int = 90,
    seed: int = 42,
) -> pd.DataFrame:
    """Deterministic (seeded) synthetic transaction set - NOT real data, a fixture
    so training/tests/CI are green offline. Fraud rows get a shifted-but-overlapping
    feature distribution (heavier-tailed amounts, more likely at odd night hours,
    concentrated in a few categories, a bumped same-day transaction count) so the
    model has genuine, learnable signal without the classes being trivially
    separable - a PR-AUC of 1.0 here would be a red flag, not a win. See
    docs/model_card.md for why that distinction matters.
    """
    rng = np.random.default_rng(seed)

    n_fraud = max(1, int(round(n_rows * fraud_rate)))
    is_fraud = np.zeros(n_rows, dtype=int)
    is_fraud[rng.choice(n_rows, size=n_fraud, replace=False)] = 1
    fraud_mask = is_fraud == 1

    entity_ids = rng.integers(0, n_entities, size=n_rows)

    # timestamps: uniform across the window, then pull a larger share of fraud
    # rows (40% vs 3% for legit) into a 00:00-05:00 slot on the same day they
    # already landed on - a mild, non-deterministic hour-of-day signal.
    minute_range = days * 24 * 60
    offsets = rng.uniform(0, minute_range, size=n_rows)
    night_prob = np.where(fraud_mask, 0.40, 0.03)
    pull_to_night = rng.random(n_rows) < night_prob
    day_index = (offsets // (24 * 60)).astype(int)
    night_minute = rng.uniform(0, 5 * 60, size=n_rows)
    offsets = np.where(pull_to_night, day_index * 24 * 60 + night_minute, offsets)
    start = pd.Timestamp("2026-01-01")
    timestamps = start + pd.to_timedelta(offsets, unit="m")

    # amount: fraud drawn from a heavier-tailed, higher-mean lognormal, heavily
    # overlapping with the legit distribution (not a clean separator by itself).
    amount = np.where(
        fraud_mask,
        rng.lognormal(mean=4.3, sigma=1.3, size=n_rows),
        rng.lognormal(mean=3.2, sigma=0.9, size=n_rows),
    )

    # category: fraud skews toward electronics/travel/online, legit toward
    # everyday spend - again overlapping, not a clean separator.
    category = np.where(
        fraud_mask,
        rng.choice(_CATEGORIES, size=n_rows, p=_CATEGORY_P_FRAUD),
        rng.choice(_CATEGORIES, size=n_rows, p=_CATEGORY_P_LEGIT),
    )

    df = pd.DataFrame(
        {
            "transaction_id": [f"tx_{i:07d}" for i in range(n_rows)],
            "timestamp": timestamps,
            "entity_id": [f"acct_{e:04d}" for e in entity_ids],
            "amount": np.round(amount, 2),
            "category": category,
            "is_fraud": is_fraud,
        }
    ).sort_values("timestamp").reset_index(drop=True)

    # velocity feature: how many transactions this entity already made the same
    # calendar day, before this one. Fraud rows get a synthetic burst added on
    # top (models card-testing / rapid-fire fraud), on the same rng stream so
    # the whole generator stays deterministic for a fixed seed.
    day = df["timestamp"].dt.floor("D")
    df[f"{FEATURE_PREFIX}entity_daily_tx_count"] = df.groupby(["entity_id", day]).cumcount()
    fraud_rows = df.index[df["is_fraud"] == 1]
    if len(fraud_rows):
        bump = rng.integers(1, 4, size=len(fraud_rows))
        df.loc[fraud_rows, f"{FEATURE_PREFIX}entity_daily_tx_count"] += bump

    return _finalize(df, "synthetic")


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #

def load(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Real data if present in `data_dir`, else the deterministic synthetic
    fallback. `df.attrs["source"]` records which one was used - always propagate
    that into any reported metrics (see scripts/train.py) so numbers are never
    silently presented as real when they're a synthetic fixture."""
    real = _detect_real(data_dir)
    if real is not None:
        return real
    return make_synthetic()


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour_of_day / day_of_week from timestamp. Applied uniformly to every
    source (real or synthetic) so the model always gets the same base feature set."""
    df = df.copy()
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    return df


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(numeric_columns, categorical_columns) present in this frame. entity_id is
    deliberately excluded: it is the split's GROUP key, not a model feature."""
    numeric = ["amount", "hour_of_day", "day_of_week"]
    numeric += sorted(c for c in df.columns if c.startswith(FEATURE_PREFIX))
    categorical = ["category"] if "category" in df.columns and df["category"].notna().any() else []
    return numeric, categorical


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Fit-ready ColumnTransformer for this frame's columns. Call `.fit_transform`
    on the TRAIN split only, then `.transform` on test - never the other way round."""
    numeric, categorical = feature_columns(df)
    transformers = []
    if categorical:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        transformers.append(("cat", encoder, categorical))
    transformers.append(("num", "passthrough", numeric))
    return ColumnTransformer(transformers=transformers)


def temporal_grouped_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leakage-safe split: every row for a given entity_id goes ENTIRELY to train
    or ENTIRELY to test (one customer/card's history never straddles the
    boundary), and entities are assigned in ascending first-seen order so the
    split is also approximately temporal - every train entity is first seen no
    later than every test entity. This is the split behind every metric this
    repo reports; any resampling or reweighting must be fit AFTER this split, on
    the train rows only (see model.py - this baseline uses balanced sample
    weights, computed from the train split's own class counts, specifically to
    avoid the classic mistake of resampling before splitting).
    """
    first_seen = df.groupby("entity_id")["timestamp"].min().sort_values()
    sizes = df.groupby("entity_id").size()

    train_budget = len(df) * (1 - test_size)
    running = 0
    train_entities = []
    for entity in first_seen.index:
        if running >= train_budget:
            break
        train_entities.append(entity)
        running += int(sizes[entity])
    train_entities = set(train_entities)

    is_train = df["entity_id"].isin(train_entities)
    train_df = df.loc[is_train].reset_index(drop=True)
    test_df = df.loc[~is_train].reset_index(drop=True)
    return train_df, test_df
