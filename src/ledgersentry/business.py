"""
The reject knob in the units a fraud desk runs on: alerts, review load, fraud
caught and missed, by count and by transaction amount.

The gap this closes: the README leads with PR-AUC and "89.8% precise", which are
model metrics. A fraud-operations manager asks different questions. How many
alerts land per 10,000 transactions? How many of them are wrong? How big is the
review queue? How much fraud, by value, gets through? Every one of those is a
count on the held-out fold, so this pipeline counts them and commits the table.

What it does NOT do: put a price on anything. expected_cost_curve_<source>.json
already prices the knob, under costs its own artifact labels ILLUSTRATIVE, and
none of those dollars appear here. The only money-like numbers in this file are
sums of the dataset's own Amount column over real transactions in the fold. No
external average-loss figure is used, because a per-transaction loss number
taken from an industry report would describe some other issuer's portfolio, not
these 56,961 transactions, and the real amounts are already in the data.

Three named policies on the same fold, so "reduced versus what?" has an answer:

  A. single_threshold_0.5  full automation. Flag p >= 0.5, clear the rest. No
     human in the loop. This is the model as a plain classifier.
  B. single_threshold_high flag p >= t, clear the rest. Same auto-flags as C,
     no review queue. What you get by just raising the threshold.
  C. review_band           the shipped knob at threshold t. Flag p >= t, clear
     p <= 1 - t, send everything in between to a human.

B and C raise the same alerts; the difference is only what happens to the
uncertain middle. B clears it and eats the frauds inside it; C pays for a
review queue to surface them. That is the trade the knob sells, and the table
states both sides of it.

One identity worth knowing before an interview: the set C surfaces (auto-flag
plus review) is exactly the set a single threshold at p > 1 - t would flag. The
knob does not find more fraud than a low single threshold does. What it adds is
a split of that set into a small tier precise enough to act on without a human
and a larger tier that goes to one. `surfaced_equals_single_low_threshold`
checks this on the real fold and records the result.

Consistency guard: the scores here are re-derived with train.py's exact split,
seed and model, and the per-threshold counts must match the committed
metrics_<source>.json row for row before anything is written. If they ever
disagree, this table describes some other model, and the run fails.

Uncertainty: the fold has 75 frauds, so the comparison carries percentile
bootstrap intervals (same method and limits as bootstrap.py: sampling noise
only, rows treated as exchangeable, so read the widths as a floor).

Run:  python scripts/business_case.py           -> artifacts/business_case_<source>.json
      python scripts/business_case.py --verify  regenerate and require a byte match
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from .config import Settings, get_settings
from .data import build_preprocessor, engineer_time_features, load, temporal_grouped_split
from .model import FraudDetector

# The operating point the README headline quotes (89.8% precise with 7.1% of
# traffic sent to review). Named here once so every sentence built from this
# artifact refers to the same row.
OPERATING_THRESHOLD = 0.95

# A fine sweep for the sensitivity table. Above ~0.95 the uncalibrated score
# saturates and coverage collapses (documented in the README); the sweep goes to
# 0.99 so that cliff is visible in the table rather than cut off.
THRESHOLD_SWEEP: tuple[float, ...] = tuple(round(0.5 + 0.01 * i, 2) for i in range(50))

PER = 10_000  # rates are per 10,000 transactions
N_RESAMPLES = 1000
CI_PERCENTILES = (2.5, 97.5)


# --------------------------------------------------------------------------- #
# pure counting (tested on synthetic arrays in tests/test_business.py)
# --------------------------------------------------------------------------- #

def _share(num: float, den: float) -> float | None:
    return round(float(num) / float(den), 4) if den else None


def policy_outcome(
    flagged: np.ndarray,
    review: np.ndarray,
    y: np.ndarray,
    amount: np.ndarray,
    hours: float,
) -> dict[str, Any]:
    """Count one policy's outcome. `flagged` and `review` are disjoint boolean
    masks; everything in neither is auto-cleared as legit. Every fraud lands in
    exactly one of caught / in-review / missed, and the counts say so."""
    flagged = np.asarray(flagged, dtype=bool)
    review = np.asarray(review, dtype=bool)
    if (flagged & review).any():
        raise ValueError("a transaction cannot be both auto-flagged and sent to review")
    y = np.asarray(y).astype(int)
    amount = np.asarray(amount, dtype=float)
    is_fraud = y == 1
    cleared = ~flagged & ~review
    n = len(y)

    n_alerts = int(flagged.sum())
    true_alerts = int((flagged & is_fraud).sum())
    false_alerts = n_alerts - true_alerts
    n_review = int(review.sum())
    fraud_in_review = int((review & is_fraud).sum())
    fraud_missed = int((cleared & is_fraud).sum())
    n_fraud = int(is_fraud.sum())

    fraud_amt = float(amount[is_fraud].sum())
    amt_caught = float(amount[flagged & is_fraud].sum())
    amt_review = float(amount[review & is_fraud].sum())
    amt_missed = float(amount[cleared & is_fraud].sum())
    legit_amt_blocked = float(amount[flagged & ~is_fraud].sum())

    def per(k: int) -> float:
        return round(k * PER / n, 1) if n else 0.0

    def per_hour(k: int) -> float | None:
        return round(k / hours, 1) if hours > 0 else None

    return {
        "counts": {
            "n_transactions": n,
            "alerts_auto_flagged": n_alerts,
            "true_alerts": true_alerts,
            "false_alerts": false_alerts,
            "sent_to_review": n_review,
            "fraud_caught_auto": true_alerts,
            "fraud_in_review": fraud_in_review,
            "fraud_missed": fraud_missed,
            "fraud_total": n_fraud,
        },
        "per_10k_transactions": {
            "alerts": per(n_alerts),
            "false_alerts": per(false_alerts),
            "review_queue": per(n_review),
            "fraud_missed": per(fraud_missed),
        },
        "per_hour_of_fold": {
            "alerts": per_hour(n_alerts),
            "false_alerts": per_hour(false_alerts),
            "review_queue": per_hour(n_review),
        },
        "rates": {
            "precision_on_alerts": _share(true_alerts, n_alerts),
            "false_alerts_per_true_alert": (
                round(false_alerts / true_alerts, 3) if true_alerts else None
            ),
            "fraud_share_caught_auto": _share(true_alerts, n_fraud),
            "fraud_share_surfaced": _share(true_alerts + fraud_in_review, n_fraud),
            "fraud_share_missed": _share(fraud_missed, n_fraud),
            "review_queue_fraud_rate": _share(fraud_in_review, n_review),
        },
        "amount": {
            "fraud_amount_total": round(fraud_amt, 2),
            "fraud_amount_caught_auto": round(amt_caught, 2),
            "fraud_amount_in_review": round(amt_review, 2),
            "fraud_amount_missed": round(amt_missed, 2),
            "legit_amount_auto_flagged": round(legit_amt_blocked, 2),
            "fraud_amount_share_caught_auto": _share(amt_caught, fraud_amt),
            "fraud_amount_share_surfaced": _share(amt_caught + amt_review, fraud_amt),
            "fraud_amount_share_missed": _share(amt_missed, fraud_amt),
        },
    }


def masks_review_band(p: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """The shipped knob (model.curve_from_scores): decide when max(p, 1-p) >= t,
    flag the decided rows with p >= 0.5, send the rest to review."""
    p = np.asarray(p, dtype=float)
    covered = np.maximum(p, 1 - p) >= t
    return covered & (p >= 0.5), ~covered


def masks_single_threshold(p: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """A plain classifier at threshold t: flag p >= t, clear everything else."""
    p = np.asarray(p, dtype=float)
    return p >= t, np.zeros(len(p), dtype=bool)


def sensitivity_table(
    p: np.ndarray,
    y: np.ndarray,
    amount: np.ndarray,
    hours: float,
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    """The review-band policy at every threshold in the sweep, flattened to the
    columns a non-ML reader scans: alerts, false alerts, review load, misses,
    share of fraud value caught."""
    rows = []
    for t in thresholds:
        o = policy_outcome(*masks_review_band(p, t), y, amount, hours)
        c, r10, rt, am = o["counts"], o["per_10k_transactions"], o["rates"], o["amount"]
        rows.append(
            {
                "review_threshold": round(float(t), 2),
                "alerts_per_10k": r10["alerts"],
                "false_alerts_per_10k": r10["false_alerts"],
                "review_queue_per_10k": r10["review_queue"],
                "review_share_of_traffic": _share(c["sent_to_review"], c["n_transactions"]),
                "precision_on_alerts": rt["precision_on_alerts"],
                "fraud_caught_auto": c["fraud_caught_auto"],
                "fraud_in_review": c["fraud_in_review"],
                "fraud_missed": c["fraud_missed"],
                "fraud_amount_share_caught_auto": am["fraud_amount_share_caught_auto"],
                "fraud_amount_share_surfaced": am["fraud_amount_share_surfaced"],
                "fraud_amount_share_missed": am["fraud_amount_share_missed"],
            }
        )
    return rows


def compare_policies(
    p: np.ndarray, y: np.ndarray, amount: np.ndarray, hours: float, t: float
) -> dict[str, Any]:
    """Policies A, B and C (module docstring) on the same fold, plus the deltas a
    manager reads: C against A (what the knob changes versus full automation)
    and C against B (what the review queue buys over just raising the bar)."""
    a = policy_outcome(*masks_single_threshold(p, 0.5), y, amount, hours)
    b = policy_outcome(*masks_single_threshold(p, t), y, amount, hours)
    c = policy_outcome(*masks_review_band(p, t), y, amount, hours)

    fa_a, fa_c = a["counts"]["false_alerts"], c["counts"]["false_alerts"]
    low_flag = np.asarray(p, dtype=float) > 1 - t
    c_flag, c_rev = masks_review_band(p, t)
    return {
        "A_single_threshold_0.5": a,
        f"B_single_threshold_{t}": b,
        f"C_review_band_{t}": c,
        "C_vs_A": {
            "false_alerts": [fa_a, fa_c],
            "false_alert_reduction_share": _share(fa_a - fa_c, fa_a),
            "fraud_missed": [a["counts"]["fraud_missed"], c["counts"]["fraud_missed"]],
            "added_review_queue": c["counts"]["sent_to_review"],
            "added_review_queue_per_10k": c["per_10k_transactions"]["review_queue"],
        },
        "C_vs_B": {
            "same_alerts": (
                b["counts"]["alerts_auto_flagged"] == c["counts"]["alerts_auto_flagged"]
            ),
            "fraud_missed": [b["counts"]["fraud_missed"], c["counts"]["fraud_missed"]],
            "fraud_amount_missed": [
                b["amount"]["fraud_amount_missed"], c["amount"]["fraud_amount_missed"]
            ],
            "cost_review_queue": c["counts"]["sent_to_review"],
        },
        "surfaced_equals_single_low_threshold": bool(
            np.array_equal(c_flag | c_rev, low_flag)
        ),
    }


def _pct(samples: list[float]) -> dict[str, float]:
    arr = np.asarray(samples, dtype=float)
    lo, hi = (float(v) for v in np.percentile(arr, CI_PERCENTILES))
    return {"ci_lower": round(lo, 4), "ci_upper": round(hi, 4)}


def bootstrap_comparison(
    p: np.ndarray, y: np.ndarray, amount: np.ndarray, t: float,
    n_resamples: int, seed: int,
) -> dict[str, Any]:
    """Percentile-bootstrap intervals on the numbers the README sentences use.
    Resamples rows with replacement, same as bootstrap.py."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y).astype(int)
    amount = np.asarray(amount, dtype=float)
    a_flag = p >= 0.5
    c_flag, c_rev = masks_review_band(p, t)
    rng = np.random.default_rng(seed)
    n = len(y)
    keys = (
        "A_false_alerts_per_10k", "C_false_alerts_per_10k", "false_alert_reduction_share",
        "C_review_queue_per_10k", "C_precision_on_alerts",
        "A_fraud_share_missed", "C_fraud_share_missed",
        "A_fraud_amount_share_missed", "C_fraud_amount_share_missed",
        "C_fraud_amount_share_caught_auto", "C_fraud_amount_share_surfaced",
    )
    out: dict[str, list[float]] = {k: [] for k in keys}
    degenerate = 0
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yb, ab = y[idx] == 1, amount[idx]
        af, cf, cr = a_flag[idx], c_flag[idx], c_rev[idx]
        n_fraud = int(yb.sum())
        fa_a = int((af & ~yb).sum())
        fa_c = int((cf & ~yb).sum())
        n_cf = int(cf.sum())
        if n_fraud == 0 or fa_a == 0 or n_cf == 0:
            degenerate += 1
            continue
        fraud_amt = float(ab[yb].sum())
        out["A_false_alerts_per_10k"].append(fa_a * PER / n)
        out["C_false_alerts_per_10k"].append(fa_c * PER / n)
        out["false_alert_reduction_share"].append((fa_a - fa_c) / fa_a)
        out["C_review_queue_per_10k"].append(int(cr.sum()) * PER / n)
        out["C_precision_on_alerts"].append(int((cf & yb).sum()) / n_cf)
        out["A_fraud_share_missed"].append(int((~af & yb).sum()) / n_fraud)
        c_missed = ~cf & ~cr & yb
        out["C_fraud_share_missed"].append(int(c_missed.sum()) / n_fraud)
        out["A_fraud_amount_share_missed"].append(float(ab[~af & yb].sum()) / fraud_amt)
        out["C_fraud_amount_share_missed"].append(float(ab[c_missed].sum()) / fraud_amt)
        out["C_fraud_amount_share_caught_auto"].append(float(ab[cf & yb].sum()) / fraud_amt)
        out["C_fraud_amount_share_surfaced"].append(
            float(ab[(cf | cr) & yb].sum()) / fraud_amt
        )
    return {
        "method": (
            f"percentile bootstrap, {n_resamples} row resamples of the test fold, "
            f"seed {seed}; sampling noise only (see bootstrap.py for the limits)"
        ),
        "n_degenerate_resamples": degenerate,
        "intervals_95": {k: _pct(v) for k, v in out.items()},
    }


# --------------------------------------------------------------------------- #
# the real fold
# --------------------------------------------------------------------------- #

def holdout_fold(cfg: Settings) -> tuple[str, pd.DataFrame, np.ndarray]:
    """(source, test_df, p_fraud) from the exact split, seed and model train.py
    uses, so these are the committed headline's own predictions."""
    df = load(data_dir=cfg.data_dir)
    source = str(df.attrs.get("source", "unknown"))
    df = engineer_time_features(df)
    train_df, test_df = temporal_grouped_split(df, test_size=cfg.test_size)
    pre = build_preprocessor(train_df)
    model = FraudDetector(
        random_state=cfg.random_state,
        max_iter=cfg.max_iter,
        learning_rate=cfg.learning_rate,
        model=cfg.model,
    ).fit(pre.fit_transform(train_df), train_df["is_fraud"].to_numpy())
    return source, test_df, model.predict_proba_fraud(pre.transform(test_df))


def check_against_committed_metrics(
    cfg: Settings, source: str, p: np.ndarray, y: np.ndarray
) -> str:
    """Fail unless the review-band counts match the committed metrics file at
    every threshold both files share."""
    path = cfg.artifact_dir / f"metrics_{source}.json"
    if not path.exists():
        return f"no {path.name} to check against"
    committed = json.loads(path.read_text())
    n_checked = 0
    for row in committed.get("coverage_precision_curve", []):
        t = float(row["review_threshold"])
        flag, rev = masks_review_band(p, t)
        mine = {
            "n_sent_to_review": int(rev.sum()),
            "n_flagged_fraud": int(flag.sum()),
            "fraud_caught_auto": int((flag & (y == 1)).sum()),
            "fraud_in_review_queue": int((rev & (y == 1)).sum()),
            "fraud_missed": int((~flag & ~rev & (y == 1)).sum()),
        }
        theirs = {k: row[k] for k in mine}
        if mine != theirs:
            raise SystemExit(
                f"FAIL: at threshold {t} this run counts {mine} but {path.name} "
                f"says {theirs}. The table would describe a different model."
            )
        n_checked += 1
    return f"counts match {path.name} at all {n_checked} committed thresholds"


def headline_sentences(cmp: dict[str, Any], boot: dict[str, Any], t: float) -> list[str]:
    """The README sentences, built from the artifact's own numbers so a sentence
    cannot drift from the file it cites."""
    a = cmp["A_single_threshold_0.5"]
    b = cmp[f"B_single_threshold_{t}"]
    c = cmp[f"C_review_band_{t}"]
    ci = boot["intervals_95"]

    def pct(x: float) -> str:
        return f"{100 * x:.1f}%"

    red = cmp["C_vs_A"]["false_alert_reduction_share"]
    red_ci = ci["false_alert_reduction_share"]
    return [
        (
            f"Versus flagging at 0.5 with no review, the {t} review band cuts false fraud "
            f"alerts from {a['per_10k_transactions']['false_alerts']} to "
            f"{c['per_10k_transactions']['false_alerts']} per 10,000 transactions "
            f"({pct(red)} fewer, 95% CI {pct(red_ci['ci_lower'])} to "
            f"{pct(red_ci['ci_upper'])}), and missed frauds go from "
            f"{a['counts']['fraud_missed']} to {c['counts']['fraud_missed']} of "
            f"{c['counts']['fraud_total']}, at the price of a review queue of "
            f"{c['per_10k_transactions']['review_queue']:.0f} per 10,000."
        ),
        (
            f"By value, the band auto-flags {pct(c['amount']['fraud_amount_share_caught_auto'])} "
            f"of the fold's fraud amount and surfaces "
            f"{pct(c['amount']['fraud_amount_share_surfaced'])} of it (auto-flag plus review); "
            f"{pct(c['amount']['fraud_amount_share_missed'])} is cleared unseen, against "
            f"{pct(a['amount']['fraud_amount_share_missed'])} with no review."
        ),
        (
            f"Raising the threshold to {t} without a review queue gives the same "
            f"{c['counts']['alerts_auto_flagged']} alerts but misses "
            f"{b['counts']['fraud_missed']} frauds instead of "
            f"{c['counts']['fraud_missed']}: the queue is what buys the high "
            f"precision without the misses."
        ),
    ]


def build_report(cfg: Settings) -> dict[str, Any]:
    source, test_df, p = holdout_fold(cfg)
    y = test_df["is_fraud"].to_numpy().astype(int)
    amount = test_df["amount"].to_numpy(dtype=float)
    ts = pd.to_datetime(test_df["timestamp"])
    hours = float((ts.max() - ts.min()).total_seconds() / 3600)

    consistency = check_against_committed_metrics(cfg, source, p, y)
    print(f"[check] {consistency}")

    t = OPERATING_THRESHOLD
    cmp = compare_policies(p, y, amount, hours, t)
    boot = bootstrap_comparison(p, y, amount, t, N_RESAMPLES, cfg.random_state)
    return {
        "data_source": source,
        "is_synthetic": source == "synthetic",
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "what": (
            "the reject knob as an operations table: alerts, false alerts, review "
            "load, frauds caught / in review / missed, by count and by the dataset's "
            "own Amount column, on the same held-out fold and model as the headline"
        ),
        "no_prices": (
            "nothing here is priced. Amounts are sums of the dataset's Amount column "
            "over real fold transactions; the ULB documentation describes Amount only "
            "as the transaction amount, so amounts are reported in the dataset's own "
            "units and as shares. No external average-loss figure is used, and nothing "
            "is taken from expected_cost_curve (whose costs are illustrative)."
        ),
        "consistency_check": consistency,
        "fold": {
            "n_transactions": int(len(y)),
            "n_fraud": int(y.sum()),
            "hours_covered": round(hours, 2),
            "fraud_amount_total": round(float(amount[y == 1].sum()), 2),
        },
        "operating_threshold": t,
        "policies": cmp,
        "bootstrap": boot,
        "sensitivity_review_band": sensitivity_table(p, y, amount, hours, THRESHOLD_SWEEP),
        "headline_sentences": headline_sentences(cmp, boot, t),
    }


def _committed_bytes(rel: str) -> bytes | None:
    root = get_settings().artifact_dir.parent
    try:
        return subprocess.run(
            ["git", "show", f"HEAD:{rel}"], cwd=root, capture_output=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LedgerSentry business-outcome table.")
    ap.add_argument("--verify", action="store_true",
                    help="regenerate and fail unless byte-identical to the committed file")
    args = ap.parse_args(argv)

    cfg = get_settings()
    report = build_report(cfg)
    payload = json.dumps(report, indent=2) + "\n"
    name = f"business_case_{report['data_source']}.json"
    out = cfg.artifact_dir / name

    for s in report["headline_sentences"]:
        print(f"[line ] {s}")

    if args.verify:
        committed = _committed_bytes(f"artifacts/{name}")
        if committed is None:
            print(f"FAIL: artifacts/{name} is not committed")
            return 1
        if committed.replace(b"\r\n", b"\n") != payload.encode():
            print(f"FAIL: regenerated {name} differs from the committed file")
            return 1
        print(f"PASS: regenerated {name} is byte-identical to the committed file")
        return 0

    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(payload, newline="\n")
    print(f"[save ] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
