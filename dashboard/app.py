"""
LedgerSentry live dashboard.

The centerpiece is the review-threshold slider: move it and the coverage-vs-
precision tradeoff recomputes LIVE from the trained model on the held-out test
split. Below it: real measured per-row latency/throughput (same
ledgersentry.stream logic the CLI replay uses) and a feed of the transactions
flagged "fraud" or sent to "review" at the current threshold.

Run:  streamlit run dashboard/app.py
The sys.path shim below lets a bare `streamlit run dashboard/app.py` work
whether or not PYTHONPATH is set.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Make the src/ package importable whether or not PYTHONPATH is set.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ledgersentry.stream import classify_stream, load_bundle, load_stream  # noqa: E402

st.set_page_config(page_title="LedgerSentry", layout="wide")


@st.cache_resource
def get_bundle() -> dict:
    return load_bundle()


@st.cache_data(show_spinner=False)
def get_full_stream():
    """The full held-out test split - loaded, split, and time-featured once
    and cached; the sidebar slider below just takes a prefix of it."""
    return load_stream(n=0)


def operating_point(df: pd.DataFrame, threshold: float) -> dict:
    """Recompute coverage/precision LIVE from the model at the slider value."""
    bundle = get_bundle()
    X = bundle["preprocessor"].transform(df)
    y = df["is_fraud"].to_numpy()
    return bundle["model"].coverage_precision_curve(X, y, [threshold])[0]


@st.cache_data(show_spinner=False)
def full_curve(n_rows: int) -> pd.DataFrame:
    """Coverage/precision across a fine threshold grid (cached backdrop chart)."""
    df, _source = get_full_stream()
    df = df.iloc[:n_rows]
    bundle = get_bundle()
    X = bundle["preprocessor"].transform(df)
    y = df["is_fraud"].to_numpy()
    grid = [round(float(t), 3) for t in np.linspace(0.0, 1.0, 21)]
    rows = bundle["model"].coverage_precision_curve(X, y, grid)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def run_stream(n_rows: int, review_threshold: float):
    """Replay the (prefix of the) test split through the same per-row-timed
    logic as `python -m ledgersentry.stream`."""
    bundle = get_bundle()
    df, _source = get_full_stream()
    df = df.iloc[:n_rows]
    t0 = time.perf_counter()
    alerts, latencies, summary = classify_stream(bundle, df, review_threshold)
    wall = time.perf_counter() - t0
    summary["throughput"] = summary["n_rows"] / wall if wall > 0 else float("nan")
    summary["wall_s"] = wall
    return alerts, latencies.tolist(), summary


st.title("LedgerSentry - live fraud detection")
st.caption(
    "Reject-to-review fraud classifier: scores each transaction and abstains "
    "('review') instead of guessing when confidence is below the slider."
)

try:
    get_bundle()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

df_full, source = get_full_stream()
if source == "synthetic":
    st.info(
        "Data source: deterministic SYNTHETIC transaction fixture (seeded, ~1% "
        "fraud) - not real financial data. See README 'Data note'. Drop a real "
        "dataset in data/ (see README) to replay that instead."
    )
else:
    st.info(f"Data source: **{source}** (real dataset dropped in data/).")

max_n = len(df_full)
step = max(1, max_n // 20)
slider_options = sorted({max(step, max_n // 4), max_n // 2, (3 * max_n) // 4, max_n})

with st.sidebar:
    st.header("Controls")
    n = st.select_slider("Transactions replayed", options=slider_options, value=max_n)
    threshold = st.slider(
        "Review threshold (abstain below this confidence)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
    )
    st.caption(
        "Raise it and the model answers fewer transactions (lower coverage) but "
        "is more precise on the fraud it does flag - that tradeoff is the whole "
        "point of the reject-to-review knob."
    )

# --- Coverage vs precision (recomputed live at the slider value) ---
st.subheader("Coverage vs precision")
sample_df = df_full.iloc[:n]
point = operating_point(sample_df, threshold)
precision = point["precision_on_flagged"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Review threshold", f"{threshold:.2f}")
c2.metric("Coverage", f"{point['coverage'] * 100:.1f}%", help="fraction decided automatically")
c3.metric(
    "Precision on flagged",
    f"{precision * 100:.1f}%" if precision is not None else "n/a",
    help="of what's flagged as fraud, fraction that truly is",
)
c4.metric("Sent to review", f"{point['n_sent_to_review']}")

curve_df = full_curve(n)
chart_df = (
    curve_df.dropna(subset=["precision_on_flagged"])
    .set_index("review_threshold")[["coverage", "precision_on_flagged"]]
)
st.line_chart(chart_df, height=320)
st.caption(
    f"Operating point at threshold={threshold:.2f}: coverage={point['coverage'] * 100:.1f}%, "
    + (
        f"precision on flagged={precision * 100:.1f}% ({point['n_flagged_fraud']} flagged)."
        if precision is not None
        else "nothing flagged as fraud at this threshold on this sample."
    )
)

# --- Live inference: real measured latency/throughput + alert feed ---
alerts, _latencies, summary = run_stream(n, threshold)

st.subheader("Live inference")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Throughput", f"{summary['throughput']:,.0f} rows/s")
m2.metric("Mean latency", f"{summary['mean_ms']:.3f} ms")
m3.metric("p95 latency", f"{summary['p95_ms']:.3f} ms")
m4.metric("Flagged fraud", f"{summary['counts']['fraud']}")
st.caption(
    f"Measured on this machine: {summary['n_rows']} transactions, single-thread, "
    f"preprocess+decide per row, {summary['wall_s']:.2f}s wall."
)

st.subheader("Alert / review feed")
if alerts:
    feed = pd.DataFrame(alerts)[
        [
            "timestamp",
            "transaction_id",
            "entity_id",
            "amount",
            "category",
            "decision",
            "p_fraud",
            "confidence",
            "true_is_fraud",
        ]
    ]
    st.dataframe(feed, use_container_width=True, hide_index=True, height=320)
else:
    st.info("Nothing flagged as fraud or sent to review at this threshold.")
