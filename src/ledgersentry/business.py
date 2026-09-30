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

Four named policies on the same fold, so "reduced versus what?" has an answer:

  A. single_threshold_0.5    full automation. Flag p >= 0.5, clear the rest. The
                             model as a plain classifier at its default cut.
  B. single_threshold_t      flag p >= t, clear the rest. Same auto-flags as C,
                             no review queue.
  C. review_band_t           the knob at threshold t. Flag p >= t, clear
                             p <= 1 - t, send everything in between to a human.
  D. single_threshold_1-t    flag everything C surfaces (p > 1 - t) as an alert,
                             with no second tier.

Where each effect comes from, stated so no sentence can blur it:
  * The false-alert cut from A to C comes from raising the auto-flag bar. B has
    exactly C's alerts and C's false alerts. The review queue does not remove a
    single false alert; `C_vs_B.same_false_alerts` checks this.
  * What the queue changes is the frauds cleared without a human look: B clears
    the uncertain middle and the frauds inside it, C sends them to review.
    Whether a reviewer then catches them is not measured here, so no sentence
    counts them as caught.
  * The set C surfaces (auto-flag plus review) is exactly D's alert set. The
    knob does not find more fraud than a low single threshold does; it splits
    that set into a small tier precise enough to act on without a human and a
    larger tier that goes to one. `surfaced_equals_single_low_threshold` checks
    this on the real fold.
  * "False alerts" counts auto-flagged legitimate transactions only. The
    legitimate transactions C sends to review are counted separately
    (`legit_sent_to_review`, `legit_flagged_or_queued`, `flagged_or_queued`) so the
    shift from "alert" to "review" is visible, not hidden in the definition.

The operating threshold (0.95) was read off the test-fold curve that
metrics_<source>.json commits. It was not selected on a validation slice, and
the artifact says so and prints its neighbours from the sweep.

Consistency guard: the scores here are re-derived with train.py's exact split,
seed and model, and the per-threshold counts must match the committed
metrics_<source>.json row for row before anything is written. If they ever
disagree, this table describes some other model, and the run fails.

Uncertainty: the fold has 75 frauds, so the comparison carries percentile
bootstrap intervals (same method and limits as bootstrap.py: sampling noise
only, rows treated as exchangeable, so read the widths as a floor). Differences
between two policies are bootstrapped PAIRED, on the same resample, never read
off two marginal intervals.

Byte-identity: the artifact holds no interpreter version, only the pinned
library versions from requirements.txt. `--verify` retrains and needs
data/creditcard.csv, so it runs locally, not in CI; CI instead rebuilds every
policy, sweep row and interval from the committed per-transaction scores
(tests/test_demo_data.py).

Run:  python scripts/business_case.py           -> artifacts/business_case_<source>.json
      python scripts/business_case.py --verify  regenerate and require a byte match
"""
from __future__ import annotations

import argparse
import json
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

# The operating point the README quotes (89.8% precise with 7.1% of traffic sent
# to review). Named here once so every sentence built from this artifact refers
# to the same row. Read off the test-fold curve, not validated (module docstring).
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
) -> dict[str, Any]:
    """Count one policy's outcome. `flagged` and `review` are disjoint boolean
    masks; everything in neither is auto-cleared as legit. Every fraud lands in
    exactly one of auto-flagged / in-review / cleared, and the counts say so."""
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
    legit_in_review = n_review - fraud_in_review
    fraud_missed = int((cleared & is_fraud).sum())
    n_fraud = int(is_fraud.sum())

    fraud_amt = float(amount[is_fraud].sum())
    amt_caught = float(amount[flagged & is_fraud].sum())
    amt_review = float(amount[review & is_fraud].sum())
    amt_missed = float(amount[cleared & is_fraud].sum())
    legit_amt_blocked = float(amount[flagged & ~is_fraud].sum())

    def per(k: int) -> float:
        return round(k * PER / n, 1) if n else 0.0

    return {
        "counts": {
            "n_transactions": n,
            "alerts_auto_flagged": n_alerts,
            "true_alerts": true_alerts,
            "false_alerts": false_alerts,
            "sent_to_review": n_review,
            "legit_sent_to_review": legit_in_review,
            "legit_flagged_or_queued": false_alerts + legit_in_review,
            "flagged_or_queued": n_alerts + n_review,
            "fraud_caught_auto": true_alerts,
            "fraud_in_review": fraud_in_review,
            "fraud_missed": fraud_missed,
            "fraud_total": n_fraud,
        },
        "per_10k_transactions": {
            "alerts": per(n_alerts),
            "false_alerts": per(false_alerts),
            "review_queue": per(n_review),
            "legit_flagged_or_queued": per(false_alerts + legit_in_review),
            "flagged_or_queued": per(n_alerts + n_review),
            "fraud_missed": per(fraud_missed),
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


def masks_single_low_threshold(p: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Policy D: flag everything the band at t does not auto-clear (p > 1 - t)."""
    p = np.asarray(p, dtype=float)
    return p > 1 - t, np.zeros(len(p), dtype=bool)


def sensitivity_table(
    p: np.ndarray,
    y: np.ndarray,
    amount: np.ndarray,
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    """The review-band policy at every threshold in the sweep, flattened to the
    columns a non-ML reader scans: alerts, false alerts, review load, misses,
    share of fraud value caught."""
    rows = []
    for t in thresholds:
        o = policy_outcome(*masks_review_band(p, t), y, amount)
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


def low_threshold_name(t: float) -> str:
    return f"D_single_threshold_{1 - t:.2f}"


def compare_policies(
    p: np.ndarray, y: np.ndarray, amount: np.ndarray, t: float
) -> dict[str, Any]:
    """Policies A to D (module docstring) on the same fold, plus the deltas that
    say where each effect comes from."""
    a = policy_outcome(*masks_single_threshold(p, 0.5), y, amount)
    b = policy_outcome(*masks_single_threshold(p, t), y, amount)
    c = policy_outcome(*masks_review_band(p, t), y, amount)
    d = policy_outcome(*masks_single_low_threshold(p, t), y, amount)

    ac, bc, cc, dc = a["counts"], b["counts"], c["counts"], d["counts"]
    c_flag, c_rev = masks_review_band(p, t)
    d_flag, _ = masks_single_low_threshold(p, t)
    same_surfaced = bool(np.array_equal(c_flag | c_rev, d_flag))
    return {
        "A_single_threshold_0.5": a,
        f"B_single_threshold_{t}": b,
        f"C_review_band_{t}": c,
        low_threshold_name(t): d,
        "B_vs_A": {
            "false_alerts": [ac["false_alerts"], bc["false_alerts"]],
            "false_alert_reduction_share": _share(
                ac["false_alerts"] - bc["false_alerts"], ac["false_alerts"]
            ),
            "fraud_missed": [ac["fraud_missed"], bc["fraud_missed"]],
        },
        "C_vs_A": {
            "false_alerts": [ac["false_alerts"], cc["false_alerts"]],
            "false_alert_reduction_share": _share(
                ac["false_alerts"] - cc["false_alerts"], ac["false_alerts"]
            ),
            "fraud_missed": [ac["fraud_missed"], cc["fraud_missed"]],
            "flagged_or_queued": [ac["flagged_or_queued"], cc["flagged_or_queued"]],
            "legit_flagged_or_queued": [
                ac["legit_flagged_or_queued"], cc["legit_flagged_or_queued"]
            ],
            "added_review_queue": cc["sent_to_review"],
            "added_review_queue_per_10k": c["per_10k_transactions"]["review_queue"],
        },
        "C_vs_B": {
            "same_alerts": bc["alerts_auto_flagged"] == cc["alerts_auto_flagged"],
            "same_false_alerts": bc["false_alerts"] == cc["false_alerts"],
            "fraud_missed": [bc["fraud_missed"], cc["fraud_missed"]],
            "frauds_sent_to_review_instead_of_cleared": (
                bc["fraud_missed"] - cc["fraud_missed"]
            ),
            "fraud_amount_missed": [
                b["amount"]["fraud_amount_missed"], c["amount"]["fraud_amount_missed"]
            ],
            "review_queue": cc["sent_to_review"],
        },
        "C_vs_D": {
            "same_surfaced_set": same_surfaced,
            "alerts": [dc["alerts_auto_flagged"], cc["alerts_auto_flagged"]],
            "precision_on_alerts": [
                d["rates"]["precision_on_alerts"], c["rates"]["precision_on_alerts"]
            ],
        },
        "surfaced_equals_single_low_threshold": same_surfaced,
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
    Resamples rows with replacement, same as bootstrap.py. Every `paired_` key
    is a difference computed inside one resample, so its interval is the
    interval of the difference, not two marginal intervals side by side."""
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
        "C_legit_flagged_or_queued_per_10k",
        "paired_A_minus_C_frauds_cleared_without_review",
        "paired_C_minus_A_fraud_amount_share_surfaced",
    )
    out: dict[str, list[float]] = {k: [] for k in keys}
    ratio: list[float] = []
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
        a_missed = ~af & yb
        c_missed = ~cf & ~cr & yb
        out["A_false_alerts_per_10k"].append(fa_a * PER / n)
        out["C_false_alerts_per_10k"].append(fa_c * PER / n)
        out["false_alert_reduction_share"].append((fa_a - fa_c) / fa_a)
        out["C_review_queue_per_10k"].append(int(cr.sum()) * PER / n)
        out["C_precision_on_alerts"].append(int((cf & yb).sum()) / n_cf)
        out["A_fraud_share_missed"].append(int(a_missed.sum()) / n_fraud)
        out["C_fraud_share_missed"].append(int(c_missed.sum()) / n_fraud)
        out["A_fraud_amount_share_missed"].append(float(ab[a_missed].sum()) / fraud_amt)
        out["C_fraud_amount_share_missed"].append(float(ab[c_missed].sum()) / fraud_amt)
        out["C_fraud_amount_share_caught_auto"].append(float(ab[cf & yb].sum()) / fraud_amt)
        c_surf = float(ab[(cf | cr) & yb].sum()) / fraud_amt
        out["C_fraud_amount_share_surfaced"].append(c_surf)
        legit_touched = int((cf & ~yb).sum()) + int((cr & ~yb).sum())
        out["C_legit_flagged_or_queued_per_10k"].append(legit_touched * PER / n)
        n_a_missed, n_c_missed = int(a_missed.sum()), int(c_missed.sum())
        out["paired_A_minus_C_frauds_cleared_without_review"].append(n_a_missed - n_c_missed)
        a_surf = float(ab[af & yb].sum()) / fraud_amt
        out["paired_C_minus_A_fraud_amount_share_surfaced"].append(c_surf - a_surf)
        if n_a_missed:
            ratio.append((n_a_missed - n_c_missed) / n_a_missed)
    intervals = {k: _pct(v) for k, v in out.items()}
    intervals["paired_reduction_share_of_A_frauds_cleared_without_review"] = _pct(ratio)
    return {
        "method": (
            f"percentile bootstrap, {n_resamples} row resamples of the test fold, "
            f"seed {seed}; sampling noise only (see bootstrap.py for the limits). "
            "paired_ keys are differences computed within each resample."
        ),
        "n_degenerate_resamples": degenerate,
        "intervals_95": intervals,
    }


# --------------------------------------------------------------------------- #
# the real fold
# --------------------------------------------------------------------------- #

def split_boundary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    """What the temporal split actually guarantees at its boundary, measured."""
    tr_ts = pd.to_datetime(train_df["timestamp"])
    te_ts = pd.to_datetime(test_df["timestamp"])
    tr_max, te_min = tr_ts.max(), te_ts.min()
    return {
        "train_last_timestamp": str(tr_max),
        "test_first_timestamp": str(te_min),
        "no_train_row_later_than_any_test_row": bool(tr_max <= te_min),
        "train_rows_later_than_first_test_row": int((tr_ts > te_min).sum()),
        "train_rows_at_boundary_second": int((tr_ts == te_min).sum()),
        "test_rows_at_boundary_second": int((te_ts == te_min).sum()),
    }


def holdout_fold(cfg: Settings) -> tuple[str, pd.DataFrame, np.ndarray]:
    """(source, test_df, p_fraud) from the exact split, seed and model train.py
    uses, so these are the committed headline's own predictions. The measured
    split boundary rides along in test_df.attrs["split_boundary"]."""
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
    test_df.attrs["split_boundary"] = split_boundary(train_df, test_df)
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


def pct(x: float, digits: int = 1) -> str:
    """A percentage that never prints 0% or 100% for a value strictly between."""
    s = f"{100 * x:.{digits}f}%"
    while 0 < x < 1 and s in (f"{0:.{digits}f}%", f"{100:.{digits}f}%") and digits < 4:
        digits += 1
        s = f"{100 * x:.{digits}f}%"
    return s


def threshold_selection(sweep: list[dict[str, Any]], t: float) -> dict[str, Any]:
    """How the operating threshold was chosen, and its neighbours in the sweep."""
    rows = {r["review_threshold"]: r for r in sweep}
    near = [round(t + d, 2) for d in (-0.01, 0.0, 0.01)]
    keep = ("review_share_of_traffic", "precision_on_alerts", "fraud_missed",
            "fraud_amount_share_surfaced")
    return {
        "how": (
            f"{t} was read off the test-fold curve in metrics_<source>.json. It was "
            "not selected on a validation slice, so it is a reported operating point, "
            "not a tuned one. Its neighbours from the sweep are below."
        ),
        "neighbours": {str(k): {c: rows[k][c] for c in keep} for k in near if k in rows},
    }


def _a(word: str) -> str:
    """The English article for a number written in digits (an 89.8%, a 97.2%)."""
    return "an" if word.startswith(("8", "11", "18")) else "a"


def _sentence_parts(report: dict[str, Any]) -> dict[str, Any]:
    t = report["operating_threshold"]
    pol = report["policies"]
    return {
        "t": t,
        "low": f"{1 - t:.2f}",
        "a": pol["A_single_threshold_0.5"],
        "b": pol[f"B_single_threshold_{t}"],
        "c": pol[f"C_review_band_{t}"],
        "cmp": pol,
        "ci": report["bootstrap"]["intervals_95"],
        "holdout": (
            "a temporal holdout"
            if report["split_boundary"]["no_train_row_later_than_any_test_row"]
            else "an approximately temporal holdout"
        ),
        "what_rows": (
            "synthetic transactions" if report["is_synthetic"] else "real card transactions"
        ),
    }


def headline_sentences(report: dict[str, Any]) -> list[str]:
    """The README sentences, built from the artifact's own numbers so a sentence
    cannot drift from the file it cites, and worded so each effect is credited
    to the policy change that produces it."""
    s = _sentence_parts(report)
    t, a, b, c, ci, cmp = s["t"], s["a"], s["b"], s["c"], s["ci"], s["cmp"]
    red, red_ci = cmp["B_vs_A"]["false_alert_reduction_share"], ci["false_alert_reduction_share"]
    miss_ci = ci["paired_A_minus_C_frauds_cleared_without_review"]
    val_ci = ci["paired_C_minus_A_fraud_amount_share_surfaced"]
    return [
        (
            f"Raising the auto-flag threshold from 0.5 to {t} cuts false fraud alerts from "
            f"{a['per_10k_transactions']['false_alerts']} to "
            f"{b['per_10k_transactions']['false_alerts']} per 10,000 transactions "
            f"({pct(red)} fewer, 95% CI {pct(red_ci['ci_lower'])} to "
            f"{pct(red_ci['ci_upper'])}). The review queue plays no part in that cut: "
            f"the same threshold with no review raises the identical "
            f"{b['counts']['alerts_auto_flagged']} alerts."
        ),
        (
            f"What the review band adds is on the miss side. The {t} threshold alone "
            f"clears {b['counts']['fraud_missed']} of {c['counts']['fraud_total']} frauds "
            f"without a human look; sending the {s['low']} to {t} middle to review "
            f"({c['per_10k_transactions']['review_queue']:.0f} per 10,000) cuts that to "
            f"{c['counts']['fraud_missed']}, against {a['counts']['fraud_missed']} under the "
            f"0.5 default (paired bootstrap, {miss_ci['ci_lower']:.0f} to "
            f"{miss_ci['ci_upper']:.0f} fewer than 0.5). Whether reviewers catch the "
            f"{c['counts']['fraud_in_review']} frauds in the queue is not measured."
        ),
        (
            f"Counting the queue, {c['per_10k_transactions']['flagged_or_queued']} transactions "
            f"per 10,000 are flagged or queued against "
            f"{a['per_10k_transactions']['flagged_or_queued']} under the 0.5 default, and "
            f"{c['per_10k_transactions']['legit_flagged_or_queued']} of them are legitimate: "
            f"most of the false-alert cut is legitimate traffic moved from alert to review, "
            f"not removed."
        ),
        (
            f"By value, the band surfaces {pct(c['amount']['fraud_amount_share_surfaced'])} "
            f"of the fold's fraud amount (auto-flag plus review, 95% CI "
            f"{pct(ci['C_fraud_amount_share_surfaced']['ci_lower'])} to "
            f"{pct(ci['C_fraud_amount_share_surfaced']['ci_upper'])}), exactly the set a "
            f"single {s['low']} threshold flags; the 0.5 default surfaces "
            f"{pct(1 - a['amount']['fraud_amount_share_missed'])} (paired difference 95% CI "
            f"{100 * val_ci['ci_lower']:.1f} to {100 * val_ci['ci_upper']:.1f} points)."
        ),
    ]


def resume_sentences(report: dict[str, Any]) -> list[str]:
    """The exact resume lines, generated here so the wording and every number in
    them trace to this artifact. tests/test_demo_data.py holds the README to
    them."""
    s = _sentence_parts(report)
    t, a, b, c, ci, cmp = s["t"], s["a"], s["b"], s["c"], s["ci"], s["cmp"]
    red, red_ci = cmp["B_vs_A"]["false_alert_reduction_share"], ci["false_alert_reduction_share"]
    miss_ci = ci["paired_A_minus_C_frauds_cleared_without_review"]
    surf_ci = ci["C_fraud_amount_share_surfaced"]
    n = c["counts"]["n_transactions"]
    review_share = c["counts"]["sent_to_review"] / n
    prec = pct(c["rates"]["precision_on_alerts"])
    a_miss, c_miss, total = (
        a["counts"]["fraud_missed"], c["counts"]["fraud_missed"], c["counts"]["fraud_total"]
    )
    fewer = (
        f"fell from {a_miss} to {c_miss} of {total} (paired bootstrap 95% CI "
        f"{miss_ci['ci_lower']:.0f} to {miss_ci['ci_upper']:.0f} fewer)"
        if miss_ci["ci_lower"] >= 1
        else f"went from {a_miss} to {c_miss} of {total} (paired bootstrap interval "
        f"includes no change)"
    )
    return [
        (
            f"Raising the auto-flag threshold from 0.5 to {t} cut false fraud alerts "
            f"{pct(red, 0)} ({a['per_10k_transactions']['false_alerts']} to "
            f"{b['per_10k_transactions']['false_alerts']} per 10,000 transactions, 95% CI "
            f"{pct(red_ci['ci_lower'], 0)} to {pct(red_ci['ci_upper'], 0)}) on "
            f"{s['holdout']} of {n:,} {s['what_rows']}; a review band on the {s['low']} to "
            f"{t} middle routed to a human "
            f"{cmp['C_vs_B']['frauds_sent_to_review_instead_of_cleared']} of the "
            f"{b['counts']['fraud_missed']} frauds that threshold alone auto-clears, at "
            f"{c['per_10k_transactions']['review_queue']:.0f} reviews per 10,000."
        ),
        (
            f"Split the {pct(c['amount']['fraud_amount_share_surfaced'])} of fraud value that "
            f"a single {s['low']} threshold surfaces (95% CI {pct(surf_ci['ci_lower'])} to "
            f"{pct(surf_ci['ci_upper'])}) into {_a(prec)} {prec}-precise auto-flag tier and "
            f"a review tier of {pct(review_share)} of traffic; against the 0.5 default, "
            f"frauds cleared without human review {fewer}."
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
    cmp = compare_policies(p, y, amount, t)
    boot = bootstrap_comparison(p, y, amount, t, N_RESAMPLES, cfg.random_state)
    sweep = sensitivity_table(p, y, amount, THRESHOLD_SWEEP)
    report: dict[str, Any] = {
        "data_source": source,
        "is_synthetic": source == "synthetic",
        "environment": {
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
        "split_boundary": test_df.attrs["split_boundary"],
        "operating_threshold": t,
        "operating_threshold_selection": threshold_selection(sweep, t),
        "policies": cmp,
        "bootstrap": boot,
        "sensitivity_review_band": sweep,
    }
    report["headline_sentences"] = headline_sentences(report)
    report["resume_sentences"] = resume_sentences(report)
    return report


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
    for s in report["resume_sentences"]:
        print(f"[cv   ] {s}")

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
