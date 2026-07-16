"""
Committed latency/throughput benchmark for the scoring path.

Why this exists: LedgerSentry markets itself as real-time fraud scoring, and
real-time has a hard meaning here - the score sits inline with payment
authorization, where the whole authorization round-trip is budgeted in tens to
a few hundred milliseconds and the fraud check gets a small slice of it. This
repo sets itself an explicit budget - single-row scoring p99 under 10 ms on
commodity hardware (an engineering target we hold ourselves to, not an
industry-published figure) - and this harness measures against it instead of
asserting it.

What it measures, all with time.perf_counter on the machine that runs it:
  single-row  score one transaction at a time (feature dict -> decision), the
              way the /predict endpoint sees traffic. p50/p95/p99/mean + rows/s.
  batch       score the whole held-out test split in one vectorized call, the
              way a backfill or micro-batch consumer would. rows/s.

Both are run through BOTH scorer implementations (see scoring.py):
  pandas      1-row DataFrame -> fitted ColumnTransformer -> model (the
              original path, kept as the honest baseline)
  compiled    precompiled numpy transform -> same model

Results are printed and written to artifacts/benchmark.json with the exact
environment (versions, CPU, data source) so a number is never quoted without
its context. Run: python scripts/bench.py
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from .config import get_settings
from .scoring import CompiledScorer, PandasScorer, TransactionScorer
from .stream import load_bundle, load_stream

SINGLE_ROW_DEFAULT = 2000
BATCH_REPEATS = 5
P99_BUDGET_MS = 10.0  # the self-imposed single-row budget the README quotes


def _percentiles(latencies_ms: np.ndarray) -> dict[str, float]:
    return {
        "mean_ms": round(float(latencies_ms.mean()), 4),
        "p50_ms": round(float(np.percentile(latencies_ms, 50)), 4),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 4),
        "p99_ms": round(float(np.percentile(latencies_ms, 99)), 4),
    }


def bench_single_row(
    scorer: TransactionScorer, rows: list[dict[str, Any]], review_threshold: float = 0.0
) -> dict[str, Any]:
    """Score row dicts one at a time (dict -> decision), timing each call."""
    # warmup: first calls pay one-time costs (imports, sklearn dispatch caches)
    for row in rows[: min(20, len(rows))]:
        scorer.score_one(row, review_threshold)

    latencies = np.empty(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
        t0 = time.perf_counter()
        scorer.score_one(row, review_threshold)
        latencies[i] = (time.perf_counter() - t0) * 1000.0
    out = {"n_rows": len(rows), **_percentiles(latencies)}
    out["rows_per_sec"] = round(len(rows) / (latencies.sum() / 1000.0), 1)
    return out


def bench_batch(
    scorer: TransactionScorer, df: pd.DataFrame, repeats: int = BATCH_REPEATS
) -> dict[str, Any]:
    """Score the whole frame in one call, best-of-N wall time (best-of because
    we want sustained compute throughput, not OS scheduling noise)."""
    scorer.score_frame(df)  # warmup
    walls = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        scorer.score_frame(df)
        walls.append(time.perf_counter() - t0)
    best = min(walls)
    return {
        "n_rows": len(df),
        "repeats": repeats,
        "best_wall_s": round(best, 4),
        "rows_per_sec": round(len(df) / best, 1),
        "per_row_us": round(best / len(df) * 1e6, 2),
    }


def _feature_dicts(df: pd.DataFrame, scorer: TransactionScorer) -> list[dict[str, Any]]:
    """Plain feature dicts, the shape a JSON /predict request arrives in."""
    cols = [c for c in scorer.numeric_cols + scorer.categorical_cols if c in df.columns]
    return df[cols].to_dict("records")


def environment() -> dict[str, Any]:
    import joblib
    import sklearn

    return {
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "measured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def run(n_single: int = SINGLE_ROW_DEFAULT) -> dict[str, Any]:
    bundle = load_bundle()
    df, source = load_stream(n=0)  # the full held-out test split
    pandas_scorer = PandasScorer(bundle["preprocessor"], bundle["model"])
    compiled_scorer = CompiledScorer(bundle["preprocessor"], bundle["model"])

    rows = _feature_dicts(df.iloc[: min(n_single, len(df))], compiled_scorer)
    print(f"[bench] data_source={source} test_split={len(df)} rows, "
          f"single-row sample={len(rows)}")

    results: dict[str, Any] = {
        "data_source": source,
        "is_synthetic": source == "synthetic",
        "p99_budget_ms": P99_BUDGET_MS,
        "environment": environment(),
        "single_row": {},
        "batch": {},
    }
    for name, scorer in [("pandas", pandas_scorer), ("compiled", compiled_scorer)]:
        single = bench_single_row(scorer, rows)
        batch = bench_batch(scorer, df)
        results["single_row"][name] = single
        results["batch"][name] = batch
        print(
            f"[single] {name:>8}: mean={single['mean_ms']:.3f} ms  "
            f"p50={single['p50_ms']:.3f}  p95={single['p95_ms']:.3f}  "
            f"p99={single['p99_ms']:.3f} ms  ({single['rows_per_sec']:,.0f} rows/s)"
        )
        print(
            f"[batch ] {name:>8}: {batch['rows_per_sec']:,.0f} rows/s  "
            f"({batch['per_row_us']:.1f} us/row over {batch['n_rows']} rows, "
            f"best of {batch['repeats']})"
        )

    p99 = results["single_row"]["compiled"]["p99_ms"]
    verdict = "MEETS" if p99 <= P99_BUDGET_MS else "MISSES"
    results["budget_verdict"] = verdict
    print(f"[budget] compiled single-row p99 {p99:.3f} ms vs {P99_BUDGET_MS:.0f} ms "
          f"budget -> {verdict}")

    artifact_dir = get_settings().artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(results, indent=2)
    # benchmark.json is always the latest run; benchmark_<source>.json is a
    # per-source snapshot, the same convention train.py uses for metrics, so a
    # real-data benchmark and a synthetic one never silently overwrite each other.
    out_path = artifact_dir / "benchmark.json"
    out_path.write_text(payload)
    (artifact_dir / f"benchmark_{source}.json").write_text(payload)
    print(f"[save  ] {out_path}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark the scoring path.")
    ap.add_argument(
        "--n-single", type=int, default=SINGLE_ROW_DEFAULT,
        help="rows for the single-row timing sample (batch always uses the full split)",
    )
    args = ap.parse_args()
    run(n_single=args.n_single)


if __name__ == "__main__":
    main()
