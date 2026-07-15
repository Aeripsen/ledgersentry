"""
Streaming replayer: replay the held-out TEST split (produced by the same
temporal_grouped_split scripts/train.py evaluates on, already timestamp-sorted
by data.py) as a live transaction stream, one row at a time, through the
trained LedgerSentry artifact.

Same honesty rule as FlowSentry's stream.py: every latency number here is
measured with time.perf_counter() around ONE row's preprocess+decide, on
whatever machine actually ran it - nothing is estimated or invented. If a real
source (Sparkov/IEEE-CIS/ULB/FDB) is dropped in data/, this replays THAT
source's test split instead and reports df.attrs["source"] honestly - it does
not assume synthetic.

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

from .data import engineer_time_features, load, temporal_grouped_split

ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts" / "ledgersentry.joblib"


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
    """Classify one row at a time. The point of this loop is honest per-row
    timing - the deployed service sees one transaction at a time too.

    Returns (alerts, latencies_ms, summary):
      alerts       list of dicts for every row decided "fraud" or sent to "review"
      latencies_ms per-row preprocess+decide latency in ms (np.ndarray)
      summary      decision counts + latency percentiles for the whole run
    """
    pre = bundle["preprocessor"]
    model = bundle["model"]
    n_rows = len(df)
    latencies = np.empty(n_rows, dtype=float)
    alerts: list[dict] = []
    counts = {"legit": 0, "fraud": 0, "review": 0}
    has_truth = "is_fraud" in df.columns
    has_category = "category" in df.columns

    for i in range(n_rows):
        row = df.iloc[[i]]  # 1-row frame keeps column names/dtypes for the preprocessor
        t0 = time.perf_counter()
        X = pre.transform(row)
        decision, p_fraud, confidence = model.decide(X, review_threshold=review_threshold)
        latencies[i] = (time.perf_counter() - t0) * 1000.0

        label = str(decision[0])
        counts[label] += 1
        if label in ("fraud", "review"):
            alerts.append(
                {
                    "timestamp": str(row["timestamp"].iloc[0]),
                    "row_index": i,
                    "transaction_id": str(row["transaction_id"].iloc[0]),
                    "entity_id": str(row["entity_id"].iloc[0]),
                    "amount": float(row["amount"].iloc[0]),
                    "category": str(row["category"].iloc[0]) if has_category else None,
                    "decision": label,
                    "p_fraud": round(float(p_fraud[0]), 4),
                    "confidence": round(float(confidence[0]), 4),
                    "true_is_fraud": int(row["is_fraud"].iloc[0]) if has_truth else None,
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
        f"[latency] per-row preprocess+decide  "
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
