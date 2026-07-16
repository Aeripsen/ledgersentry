"""
Streaming replayer: replay the held-out TEST split (produced by the same
temporal_grouped_split scripts/train.py evaluates on, already timestamp-sorted
by data.py) as a live transaction stream, one row at a time, through the
trained LedgerSentry artifact.

Honesty rule: every latency number here is measured with time.perf_counter()
around ONE row's transform+decide, on whatever machine actually ran it -
nothing is estimated or invented. If a real source (Sparkov/IEEE-CIS/ULB/FDB)
is dropped in data/, this replays THAT source's test split instead and reports
df.attrs["source"] honestly - it does not assume synthetic.

Why the test split, not the whole set: it's the held-out portion the model
never trained on (same rows scripts/train.py's PR-AUC is measured against), so
the replay is scoring transactions the way the deployed service would see
genuinely unseen ones, not re-scoring its own training data. It's also already
in ascending timestamp order (a filtered subset of data.py's timestamp-sorted
frame stays sorted), so replaying it row 0..n IS replaying in real time order.

Run:  python -m ledgersentry.stream --n 0     (0 = replay the whole test split)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import get_settings
from .data import engineer_time_features, load, temporal_grouped_split
from .scoring import build_scorer

ARTIFACT = get_settings().artifact_dir / "ledgersentry.joblib"


def load_bundle(path: Path = ARTIFACT) -> dict:
    """Load the serving artifact dict: {preprocessor, model}."""
    if not path.exists():
        raise FileNotFoundError(
            f"model artifact missing at {path}; run `python scripts/train.py` first"
        )
    return joblib.load(path)


def load_stream(n: int = 0) -> tuple[pd.DataFrame, str]:
    """The held-out TEST split (real source if one's dropped in data/, else the
    deterministic synthetic fixture), time-featured, already timestamp-sorted.
    n <= 0 replays the whole test split; n > 0 replays the first n rows.
    Returns (df, source) - source is whatever df.attrs["source"] says
    (data.py sets this honestly, "synthetic" or a real dataset name)."""
    df = load()
    source = df.attrs.get("source", "unknown")
    df = engineer_time_features(df)
    _train_df, test_df = temporal_grouped_split(df, test_size=0.2)
    if n > 0:
        test_df = test_df.iloc[:n].reset_index(drop=True)
    return test_df, source


def classify_stream(bundle: dict, df: pd.DataFrame, review_threshold: float = 0.0):
    """Classify one row at a time through the same compiled scorer the /predict
    endpoint serves with. What's timed is exactly the serving work - transform
    one feature dict + model decide. Building the dict happens OUTSIDE the
    timer, because in the deployed service that dict arrives as the request
    body; it is not scoring work.

    Returns (alerts, latencies_ms, summary):
      alerts       list of dicts for every row decided "fraud" or sent to "review"
      latencies_ms per-row transform+decide latency in ms (np.ndarray)
      summary      decision counts + latency percentiles for the whole run
    """
    scorer = build_scorer(bundle)
    n_rows = len(df)
    feature_cols = [
        c for c in scorer.numeric_cols + scorer.categorical_cols if c in df.columns
    ]
    feature_rows = df[feature_cols].to_dict("records")
    latencies = np.empty(n_rows, dtype=float)
    alerts: list[dict] = []
    counts = {"legit": 0, "fraud": 0, "review": 0}
    has_truth = "is_fraud" in df.columns
    has_category = "category" in df.columns

    for i, features in enumerate(feature_rows):
        t0 = time.perf_counter()
        result = scorer.score_one(features, review_threshold=review_threshold)
        latencies[i] = (time.perf_counter() - t0) * 1000.0

        label = result.decision
        counts[label] += 1
        if label in ("fraud", "review"):
            row = df.iloc[i]
            alerts.append(
                {
                    "timestamp": str(row["timestamp"]),
                    "row_index": i,
                    "transaction_id": str(row["transaction_id"]),
                    "entity_id": str(row["entity_id"]),
                    "amount": float(row["amount"]),
                    "category": str(row["category"]) if has_category else None,
                    "decision": label,
                    "p_fraud": round(result.p_fraud, 4),
                    "confidence": round(result.confidence, 4),
                    "true_is_fraud": int(row["is_fraud"]) if has_truth else None,
                }
            )

    summary = {
        "n_rows": n_rows,
        "counts": counts,
        "mean_ms": float(latencies.mean()) if n_rows else float("nan"),
        "p50_ms": float(np.percentile(latencies, 50)) if n_rows else float("nan"),
        "p95_ms": float(np.percentile(latencies, 95)) if n_rows else float("nan"),
        "p99_ms": float(np.percentile(latencies, 99)) if n_rows else float("nan"),
    }
    return alerts, latencies, summary


def _format_alert(a: dict) -> str:
    truth = f" true={a['true_is_fraud']}" if a["true_is_fraud"] is not None else ""
    return (
        f"ALERT row#{a['row_index']:<5} {a['decision']:<6} "
        f"p_fraud={a['p_fraud']:.3f} conf={a['confidence']:.3f}{truth}  "
        f"tx={a['transaction_id']} amt={a['amount']:.2f} cat={a['category']}"
    )


def run(n: int, review_threshold: float, max_alerts: int) -> dict:
    bundle = load_bundle()
    df, source = load_stream(n)
    print(
        f"[stream] replaying {len(df)} transactions from the '{source}' test split "
        f"(review_threshold={review_threshold})\n"
    )

    wall_start = time.perf_counter()
    alerts, _latencies, summary = classify_stream(bundle, df, review_threshold)
    wall = time.perf_counter() - wall_start

    for a in alerts[:max_alerts]:
        print(_format_alert(a))
    if len(alerts) > max_alerts:
        print(f"... {len(alerts) - max_alerts} more alerts not shown")

    throughput = summary["n_rows"] / wall if wall > 0 else float("nan")
    c = summary["counts"]
    print("\n" + "=" * 70)
    print(
        f"[rows   ] {summary['n_rows']}   fraud={c['fraud']}  "
        f"legit={c['legit']}  review={c['review']}"
    )
    print(
        f"[latency] per-row transform+decide (compiled scorer)  "
        f"mean={summary['mean_ms']:.3f} ms  p50={summary['p50_ms']:.3f} ms  "
        f"p95={summary['p95_ms']:.3f} ms  p99={summary['p99_ms']:.3f} ms"
    )
    print(
        f"[through] {throughput:,.0f} rows/sec over {wall:.2f}s wall "
        f"(single-thread, this machine, data_source={source})"
    )
    print("=" * 70)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Replay the held-out test split as a live transaction stream "
        "through LedgerSentry."
    )
    ap.add_argument("--n", type=int, default=0, help="rows to replay (0 = whole test split)")
    ap.add_argument(
        "--review-threshold",
        type=float,
        default=0.0,
        help="below this confidence the decision becomes 'review'",
    )
    ap.add_argument(
        "--max-alerts", type=int, default=25, help="how many alerts to print inline"
    )
    args = ap.parse_args()
    run(args.n, args.review_threshold, args.max_alerts)


if __name__ == "__main__":
    main()
