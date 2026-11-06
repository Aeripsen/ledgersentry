"""
Rolling-origin (expanding-window) validation, a separate analysis beside the
single-fold headline.

What it measures that bootstrap.py cannot: the headline is one temporal fold,
so its intervals cover sampling noise inside that fold and nothing about how
much the numbers move when the cut moves. Here the model is retrained at
several successive cuts and tested on the window after each one.

Folds. `train_fractions` (default 0.2, 0.4, 0.6, 0.8) are row budgets. For each
one, the budget rule from data.temporal_grouped_split picks the first entity
(in first-seen order) that does not fit, and that entity's first-seen
timestamp becomes the cut T. Fold k then is:

    train  = every entity first seen strictly before T_k
    test   = every entity first seen in [T_k, T_{k+1})   (the last fold runs to the end)

An entity's rows all go to one side, so no entity straddles a cut. Where each
entity is one row (ULB), the cut is also a clean time cut: every train row is
earlier than every test row and no second is split. Where entities have many
rows (the synthetic fixture), a train entity's later rows can interleave in
time with test rows, the same "approximately temporal" property
temporal_grouped_split documents. When no
transaction shares the cut's second, the train side equals what
temporal_grouped_split(df, 1 - fraction) returns; on ULB that holds at 0.8, so
the last fold is the headline split row for row. Train windows only grow
(expanding window); nothing is shuffled.

A test window with fewer than `min_test_fraud` frauds is merged with the next
window (the last one with the previous), and the merge is recorded.

Two models per fold, the same separation the repo already uses:

  scoring model     fit on the whole train window, as train.py does. Gives
                    PR-AUC, recall, precision and false alerts per 10,000 at the
                    raw-score thresholds.
  calibration model fit on the first (1 - calibration_size) of the train
                    window; the Platt map is fit on the remaining, later slice;
                    calibration error is measured on the test window. As
                    calibration.py does, so the calibrator never sees rows its
                    model trained on and never sees the test window.

Intervals: a seeded percentile bootstrap for PR-AUC and calibration error
(rows resampled independently inside each window; the pooled interval
resamples inside each window and concatenates), Wilson intervals for recall,
precision and the false-alert rate. Both treat rows as exchangeable, which
bursty fraud is not, so they are a floor on the uncertainty (same caveat as
bootstrap.py).

Run: python scripts/rolling_origin.py   -> artifacts/rolling_origin_<source>.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from .bootstrap import CI_PERCENTILES, wilson_interval
from .calibration import PlattCalibrator
from .config import REPO_ROOT, get_settings
from .data import build_preprocessor, engineer_time_features, load
from .model import FraudDetector, curve_from_scores

DEFAULT_TRAIN_FRACTIONS = (0.2, 0.4, 0.6, 0.8)
MIN_TEST_FRAUD = 30
MIN_FOLDS = 4
# 0.5 is the default single cut; 0.95 is the operating point the README and the
# business case report. Both are raw-score thresholds, as in the headline.
THRESHOLDS = (0.5, 0.95)
ECE_BINS = 10


@dataclass(frozen=True)
class Fold:
    index: int  # 1-based
    cut: pd.Timestamp  # train is first seen strictly before this
    end: pd.Timestamp | None  # test is first seen before this; None = to the end
    train_mask: np.ndarray
    test_mask: np.ndarray
    merged_windows: int  # 1 unless short windows were merged into this one


def _first_seen(df: pd.DataFrame) -> pd.Series:
    """Per-row first-seen timestamp of that row's entity."""
    return df.groupby("entity_id")["timestamp"].transform("min")


def cut_timestamp(df: pd.DataFrame, train_fraction: float) -> pd.Timestamp:
    """The cut for one row budget, using temporal_grouped_split's rule: add
    entities in first-seen order until the running row count reaches the
    budget; the first entity left out sets the cut. Ties are ordered by entity
    id; they cannot change the result because the cut is a timestamp."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    first = df.groupby("entity_id")["timestamp"].min()
    sizes = df.groupby("entity_id").size()
    order = pd.DataFrame({"first": first, "size": sizes}).reset_index()
    order = order.sort_values(["first", "entity_id"], kind="mergesort")
    running = order["size"].cumsum().to_numpy()
    budget = len(df) * train_fraction
    # temporal_grouped_split adds an entity while running-before-it < budget,
    # so the first excluded entity is the first whose running-before >= budget
    running_before = running - order["size"].to_numpy()
    excluded = np.flatnonzero(running_before >= budget)
    if len(excluded) == 0:
        raise ValueError(f"train_fraction {train_fraction} leaves no test rows")
    return pd.Timestamp(order["first"].iloc[int(excluded[0])])


def split_before(df: pd.DataFrame, cut: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows whose entity is first seen before `cut`, and the rest."""
    before = (_first_seen(df) < cut).to_numpy()
    return df.loc[before].reset_index(drop=True), df.loc[~before].reset_index(drop=True)


def build_folds(
    df: pd.DataFrame,
    train_fractions: Sequence[float] = DEFAULT_TRAIN_FRACTIONS,
    min_test_fraud: int = MIN_TEST_FRAUD,
) -> list[Fold]:
    """Expanding-window folds. `df` must be the canonical frame (sorted by
    timestamp). Short windows are merged as described in the module docstring."""
    if list(train_fractions) != sorted(set(train_fractions)):
        raise ValueError("train_fractions must be strictly increasing")
    cuts = sorted({cut_timestamp(df, f) for f in train_fractions})
    first = _first_seen(df).to_numpy()
    y = df["is_fraud"].to_numpy()

    # windows as [start, end) on first-seen time, with a count of merged windows
    windows: list[list[Any]] = []
    for i, c in enumerate(cuts):
        end = cuts[i + 1] if i + 1 < len(cuts) else None
        windows.append([c, end, 1])

    def frauds(w: list[Any]) -> int:
        m = first >= np.datetime64(w[0])
        if w[1] is not None:
            m &= first < np.datetime64(w[1])
        return int(y[m].sum())

    while len(windows) > 1:
        short = [i for i, w in enumerate(windows) if frauds(w) < min_test_fraud]
        if not short:
            break
        i = short[0]
        j = i + 1 if i + 1 < len(windows) else i - 1
        a, b = sorted((i, j))
        windows[a] = [windows[a][0], windows[b][1], windows[a][2] + windows[b][2]]
        del windows[b]
    if frauds(windows[0]) < min_test_fraud:
        raise ValueError(
            f"only {frauds(windows[0])} frauds after the first cut; cannot reach "
            f"{min_test_fraud} per test window even with every window merged"
        )

    folds = []
    for k, (start, end, merged) in enumerate(windows, start=1):
        train_mask = first < np.datetime64(start)
        test_mask = first >= np.datetime64(start)
        if end is not None:
            test_mask &= first < np.datetime64(end)
        folds.append(Fold(
            index=k,
            cut=pd.Timestamp(start),
            end=pd.Timestamp(end) if end is not None else None,
            train_mask=train_mask,
            test_mask=test_mask,
            merged_windows=int(merged),
        ))
    return folds


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = ECE_BINS) -> float:
    """ECE over equal-count bins of the sorted scores: sum over bins of
    |sum(p) - sum(y)| / N, i.e. the count-weighted gap between mean predicted
    probability and observed fraud rate. Equal-count rather than equal-width
    because at a 0.17% base rate equal-width bins put nearly every row in the
    first bin (same reason calibration.py uses quantile bins)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="mergesort")
    total = 0.0
    for idx in np.array_split(order, n_bins):
        if len(idx):
            total += abs(float(p[idx].sum()) - float(y[idx].sum()))
    return total / len(y)


def _interval(lo: float, hi: float) -> dict[str, float]:
    return {"ci_lower": round(lo, 4), "ci_upper": round(hi, 4)}


def _proportion(successes: int, trials: int, digits: int = 4) -> dict[str, Any]:
    if trials == 0:
        return {"successes": 0, "trials": 0, "value": None, "ci_lower": None, "ci_upper": None}
    lo, hi = wilson_interval(successes, trials)
    return {
        "successes": int(successes),
        "trials": int(trials),
        "value": round(successes / trials, digits),
        "ci_lower": round(lo, digits),
        "ci_upper": round(hi, digits),
    }


def threshold_metrics(y: np.ndarray, p: np.ndarray, thresholds: Sequence[float]) -> list[dict]:
    """Flag rule p >= t on raw scores, with Wilson intervals. At t = 0.95 the
    same rows are the auto-flags of the symmetric review band (confidence >= t
    and p >= 0.5), so the band's queue and unseen frauds are reported too, from
    curve_from_scores, and its flag counts are checked against these."""
    y = np.asarray(y).astype(int)
    n = len(y)
    n_fraud = int(y.sum())
    curve = curve_from_scores(p, y, thresholds)
    rows = []
    for t, band in zip(thresholds, curve, strict=True):
        flagged = p >= t
        tp = int((flagged & (y == 1)).sum())
        fp = int((flagged & (y == 0)).sum())
        if band["n_flagged_fraud"] != tp + fp or band["fraud_caught_auto"] != tp:
            raise AssertionError(f"flag counts at {t} disagree with curve_from_scores")
        fa = _proportion(fp, n, digits=8)
        rows.append({
            "threshold": t,
            "n_flagged": tp + fp,
            "frauds_flagged": tp,
            "false_alerts": fp,
            "recall": _proportion(tp, n_fraud),
            "precision": _proportion(tp, tp + fp),
            "false_alerts_per_10k": {
                "value": round(fp / n * 1e4, 2),
                "ci_lower": round(fa["ci_lower"] * 1e4, 2),
                "ci_upper": round(fa["ci_upper"] * 1e4, 2),
            },
            "review_band": {
                "sent_to_review": band["n_sent_to_review"],
                "frauds_in_review": band["fraud_in_review_queue"],
                "frauds_cleared_unseen": band["fraud_missed"],
                "review_per_10k": round(band["n_sent_to_review"] / n * 1e4, 1),
            },
        })
    return rows


def _boot_summary(point: float, samples: list[float]) -> dict[str, Any]:
    lo, hi = (float(v) for v in np.percentile(samples, CI_PERCENTILES))
    return {"value": round(point, 4), **_interval(lo, hi), "ci_width": round(hi - lo, 4)}


def bootstrap_windows(
    windows: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Percentile bootstrap of PR-AUC (raw scores) and ECE (Platt scores) per
    window and pooled. Each window = (y, p_raw, p_platt). Every resample draws
    each window's rows with replacement and the pooled statistic is computed on
    the concatenation of those same draws (stratified by window). A resample in
    which a window has no fraud is counted and skipped for that window.

    Each window has its own generator. The last one is default_rng(seed), the
    generator bootstrap.py uses, so when the last window is the headline fold
    its PR-AUC interval is the committed one; the others are seeded [seed, k]."""
    rngs = [np.random.default_rng([seed, k]) for k in range(len(windows) - 1)]
    rngs.append(np.random.default_rng(seed))
    per: list[dict[str, list[float]]] = [{"pr_auc": [], "ece": []} for _ in windows]
    pooled: dict[str, list[float]] = {"pr_auc": [], "ece": []}
    degenerate = [0] * len(windows)
    for _ in range(n_resamples):
        ys, ps, cs = [], [], []
        for k, (y, p, c) in enumerate(windows):
            idx = rngs[k].integers(0, len(y), size=len(y))
            yb, pb, cb = y[idx], p[idx], c[idx]
            ys.append(yb)
            ps.append(pb)
            cs.append(cb)
            if int(yb.sum()) == 0:
                degenerate[k] += 1
                continue
            per[k]["pr_auc"].append(float(average_precision_score(yb, pb)))
            per[k]["ece"].append(expected_calibration_error(yb, cb))
        y_all, p_all, c_all = np.concatenate(ys), np.concatenate(ps), np.concatenate(cs)
        pooled["pr_auc"].append(float(average_precision_score(y_all, p_all)))
        pooled["ece"].append(expected_calibration_error(y_all, c_all))
    return {"per_window": per, "pooled": pooled, "degenerate": degenerate}


# --------------------------------------------------------------------------- #
# per-fold fitting
# --------------------------------------------------------------------------- #

def _fit_scores(
    train_df: pd.DataFrame, score_dfs: Sequence[pd.DataFrame], model_kw: dict[str, Any]
) -> list[np.ndarray]:
    pre = build_preprocessor(train_df)
    X_train = pre.fit_transform(train_df)
    model = FraudDetector(**model_kw).fit(X_train, train_df["is_fraud"].to_numpy())
    return [model.predict_proba_fraud(pre.transform(d)) for d in score_dfs]


def _ts(t: pd.Timestamp | None, origin: pd.Timestamp | None) -> dict[str, Any] | None:
    if t is None:
        return None
    out: dict[str, Any] = {"timestamp": str(t)}
    if origin is not None:
        secs = (t - origin).total_seconds()
        out["seconds_since_origin"] = int(secs)
        out["hours_since_origin"] = round(secs / 3600, 2)
    return out


def evaluate(
    df: pd.DataFrame,
    *,
    model_kw: dict[str, Any],
    train_fractions: Sequence[float] = DEFAULT_TRAIN_FRACTIONS,
    min_test_fraud: int = MIN_TEST_FRAUD,
    calibration_size: float = 0.2,
    thresholds: Sequence[float] = THRESHOLDS,
    n_resamples: int = 1000,
    seed: int = 42,
    time_origin: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Run every fold on a canonical frame with time features already added.
    Returns the report body (no file IO)."""
    folds = build_folds(df, train_fractions, min_test_fraud)
    first = _first_seen(df)

    fold_reports: list[dict[str, Any]] = []
    windows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for fold in folds:
        train_df = df.loc[fold.train_mask].reset_index(drop=True)
        test_df = df.loc[fold.test_mask].reset_index(drop=True)
        y_test = test_df["is_fraud"].to_numpy().astype(int)

        # leakage checks, recorded rather than assumed
        train_ids = set(train_df["entity_id"])
        straddling = int(test_df["entity_id"].isin(train_ids).sum())
        if straddling:
            raise AssertionError(f"fold {fold.index}: {straddling} test rows share an entity")
        if first[fold.train_mask].max() >= fold.cut:
            raise AssertionError(f"fold {fold.index}: train entity first seen after the cut")

        (p_raw,) = _fit_scores(train_df, [test_df], model_kw)

        # calibration: model on the earlier part of the train window, Platt on
        # the later part, error measured on the test window
        cal_cut = cut_timestamp(train_df, 1 - calibration_size)
        fit_df, cal_df = split_before(train_df, cal_cut)
        p_cal_slice, p_test_cal_model = _fit_scores(fit_df, [cal_df, test_df], model_kw)
        y_cal = cal_df["is_fraud"].to_numpy().astype(int)
        platt = PlattCalibrator().fit(p_cal_slice, y_cal)
        p_platt = platt.transform(p_test_cal_model)
        a, b = platt.coefficients
        cal_rate = float(y_cal.mean())

        windows.append((y_test, p_raw, p_platt))
        n_fraud = int(y_test.sum())
        base = float(y_test.mean())
        fold_reports.append({
            "fold": fold.index,
            "merged_windows": fold.merged_windows,
            "train": {
                "first_seen_before": _ts(fold.cut, time_origin),
                "n_rows": int(len(train_df)),
                "n_fraud": int(train_df["is_fraud"].sum()),
                "last_timestamp": str(train_df["timestamp"].max()),
            },
            "test": {
                "first_seen_from": _ts(fold.cut, time_origin),
                "first_seen_before": _ts(fold.end, time_origin),
                "n_rows": int(len(test_df)),
                "n_fraud": n_fraud,
                "first_timestamp": str(test_df["timestamp"].min()),
                "last_timestamp": str(test_df["timestamp"].max()),
                "hours_spanned": round(
                    (test_df["timestamp"].max() - test_df["timestamp"].min()).total_seconds()
                    / 3600, 2),
            },
            "entities_straddling_cut": straddling,
            "pr_auc_point": float(average_precision_score(y_test, p_raw)),
            "no_skill_baseline": round(base, 6),
            "thresholds": threshold_metrics(y_test, p_raw, thresholds),
            "calibration": {
                "split": (
                    "calibration model fit on train rows first seen before "
                    f"{cal_cut}; Platt map fit on the remaining train rows "
                    f"(first seen from {cal_cut} to {fold.cut}); measured on this "
                    "test window, which neither saw"
                ),
                "fit_slice": {"n_rows": int(len(fit_df)), "n_fraud": int(fit_df["is_fraud"].sum())},
                "calibration_slice": {
                    "first_seen_from": _ts(cal_cut, time_origin),
                    "n_rows": int(len(cal_df)),
                    "n_fraud": int(y_cal.sum()),
                },
                "platt_map": {"a": round(a, 6), "b": round(b, 6)},
                "ece_point": expected_calibration_error(y_test, p_platt),
                "ece_raw": round(expected_calibration_error(y_test, p_test_cal_model), 6),
                "brier": {
                    "raw": round(float(brier_score_loss(y_test, p_test_cal_model)), 6),
                    "platt": round(float(brier_score_loss(y_test, p_platt)), 6),
                    # constant prediction at the calibration slice's fraud rate:
                    # the no-skill reference that does not look at the test window
                    "constant_cal_slice_rate": round(
                        float(brier_score_loss(y_test, np.full(len(y_test), cal_rate))), 6),
                },
                "expected_frauds_platt": round(float(p_platt.sum()), 1),
                "observed_frauds": n_fraud,
            },
        })

    boot = bootstrap_windows(windows, n_resamples, seed)
    for k, rep in enumerate(fold_reports):
        pr = _boot_summary(rep.pop("pr_auc_point"), boot["per_window"][k]["pr_auc"])
        pr["lift_over_no_skill"] = round(pr["value"] / rep["no_skill_baseline"], 1)
        pr["ci_lower_lift"] = round(pr["ci_lower"] / rep["no_skill_baseline"], 1)
        rep["pr_auc"] = pr
        rep["n_degenerate_resamples"] = boot["degenerate"][k]
        cal = rep["calibration"]
        # ECE is on the order of 1e-4 here, so 6 decimals, not _boot_summary's 4
        lo, hi = np.percentile(boot["per_window"][k]["ece"], CI_PERCENTILES)
        cal["ece_platt"] = {
            "value": round(cal.pop("ece_point"), 6),
            "ci_lower": round(float(lo), 6),
            "ci_upper": round(float(hi), 6),
        }

    y_all = np.concatenate([w[0] for w in windows])
    p_all = np.concatenate([w[1] for w in windows])
    c_all = np.concatenate([w[2] for w in windows])
    pooled_pr = _boot_summary(float(average_precision_score(y_all, p_all)),
                              boot["pooled"]["pr_auc"])
    base_all = float(y_all.mean())
    pooled_pr["lift_over_no_skill"] = round(pooled_pr["value"] / base_all, 1)
    lo, hi = np.percentile(boot["pooled"]["ece"], CI_PERCENTILES)

    per_fold_pr = [r["pr_auc"]["value"] for r in fold_reports]
    pooled = {
        "n_rows": int(len(y_all)),
        "n_fraud": int(y_all.sum()),
        "no_skill_baseline": round(base_all, 6),
        "pr_auc": pooled_pr,
        "pr_auc_note": (
            "PR-AUC over the concatenated test windows. Each window is scored by "
            "its own fold's model, so this ranks scores from different models "
            "against each other; the per-fold values and their spread are the "
            "cleaner read."
        ),
        "pr_auc_across_folds": {
            "min": min(per_fold_pr),
            "max": max(per_fold_pr),
            "mean": round(float(np.mean(per_fold_pr)), 4),
        },
        "thresholds": threshold_metrics(y_all, p_all, thresholds),
        "calibration": {
            "ece_platt": {
                "value": round(expected_calibration_error(y_all, c_all), 6),
                "ci_lower": round(float(lo), 6),
                "ci_upper": round(float(hi), 6),
            },
            "expected_frauds_platt": round(float(c_all.sum()), 1),
            "observed_frauds": int(y_all.sum()),
        },
    }
    return {"folds": fold_reports, "pooled": pooled}


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def _identity_checks(report: dict[str, Any], source: str) -> dict[str, Any]:
    """On ULB the last fold is the headline split, so the scoring model must
    reproduce the committed metrics; any mismatch means this analysis is not
    measuring the shipped pipeline, and the run fails.

    The calibration side is compared and recorded, not required to match. The
    committed calibration split (temporal_grouped_split inside the train window)
    cuts through a second that 5 ULB rows share, 1 on the fit side and 4 on the
    calibration side. This analysis never splits a second, so all 5 land on the
    calibration side: one row fewer to fit, one more to calibrate on."""
    art = get_settings().artifact_dir
    metrics_path = art / f"metrics_{source}.json"
    cal_path = art / f"calibration_{source}.json"
    if source == "synthetic" or not metrics_path.exists() or not cal_path.exists():
        return {"checked": False}
    m = json.loads(metrics_path.read_text())
    c = json.loads(cal_path.read_text())
    last = report["folds"][-1]
    t05 = last["thresholds"][0]
    boot_path = art / f"bootstrap_{source}.json"
    boot_ci = json.loads(boot_path.read_text())["pr_auc"] if boot_path.exists() else None
    pairs = {
        "pr_auc_ci": (
            [last["pr_auc"]["ci_lower"], last["pr_auc"]["ci_upper"]],
            [boot_ci["ci_lower"], boot_ci["ci_upper"]] if boot_ci else None,
        ),
        "n_train": (last["train"]["n_rows"], m["n_train"]),
        "n_test": (last["test"]["n_rows"], m["n_test"]),
        "n_test_fraud": (last["test"]["n_fraud"], m["n_test_fraud"]),
        "pr_auc": (last["pr_auc"]["value"], m["pr_auc"]),
        "recall_at_0.5": (t05["recall"]["value"], m["recall_at_full_coverage"]),
    }
    bad = {k: v for k, v in pairs.items() if v[0] != v[1]}
    if bad:
        raise AssertionError(f"last fold does not reproduce the committed headline: {bad}")
    cal = last["calibration"]
    return {
        "checked": True,
        "scoring_model_matches": {
            "against": [metrics_path.name, boot_path.name],
            "values": {k: v[0] for k, v in pairs.items()},
        },
        "calibration_compared": {
            "against": cal_path.name,
            "here_vs_committed": {
                "n_fit": [cal["fit_slice"]["n_rows"], c["n_fit"]],
                "n_cal": [cal["calibration_slice"]["n_rows"], c["n_cal"]],
                "brier_test_platt": [cal["brier"]["platt"], c["brier_test"]["platt"]],
                "platt_a": [cal["platt_map"]["a"], c["platt_map"]["a"]],
                "platt_b": [cal["platt_map"]["b"], c["platt_map"]["b"]],
            },
            "why_not_identical": (
                "the committed calibration split cuts through a second shared by 5 "
                "rows (1 fit side, 4 calibration side); this analysis keeps that "
                "second whole on the calibration side"
            ),
        },
    }


def build_report(train_fractions: Sequence[float] = DEFAULT_TRAIN_FRACTIONS) -> dict[str, Any]:
    cfg = get_settings()
    df = load(data_dir=cfg.data_dir)
    source = df.attrs.get("source", "unknown")
    df = engineer_time_features(df)
    # ULB's Time column is seconds since the first transaction; the loader maps
    # it onto this origin, so cuts are also given in ULB's own Time units
    origin = pd.Timestamp("2013-01-01") if source == "ulb_creditcard" else None
    print(f"[data ] source={source} rows={len(df)} frauds={int(df['is_fraud'].sum())}")

    body = evaluate(
        df,
        model_kw={
            "random_state": cfg.random_state,
            "max_iter": cfg.max_iter,
            "learning_rate": cfg.learning_rate,
            "model": cfg.model,
        },
        train_fractions=train_fractions,
        calibration_size=cfg.calibration_size,
        n_resamples=cfg.bootstrap_resamples,
        seed=cfg.random_state,
        time_origin=origin,
    )
    report: dict[str, Any] = {
        "data_source": source,
        "is_synthetic": source == "synthetic",
        "model": cfg.model,
        "scheme": "rolling origin, expanding window",
        "train_fractions": list(train_fractions),
        "min_test_fraud": MIN_TEST_FRAUD,
        "calibration_size": cfg.calibration_size,
        "thresholds": list(THRESHOLDS),
        "threshold_rule": (
            "flag when raw P(fraud) >= threshold. At 0.95 the review_band block is "
            "the symmetric knob: flag at >= 0.95, review between 0.05 and 0.95, "
            "clear at <= 0.05."
        ),
        "intervals": {
            "pr_auc_and_ece": (
                f"percentile bootstrap, {cfg.bootstrap_resamples} resamples, seed "
                f"{cfg.random_state}, rows resampled within each test window; pooled "
                "resamples within each window and concatenates"
            ),
            "recall_precision_false_alerts": "Wilson score interval, 95%",
            "caveat": (
                "Both treat rows as exchangeable. Fraud arrives in bursts, so these "
                "intervals are a floor on the uncertainty."
            ),
        },
        "ece_definition": (
            f"{ECE_BINS} equal-count bins of the sorted calibrated scores; sum over "
            "bins of |sum(p) - sum(y)| divided by the number of rows"
        ),
        **body,
    }
    report["identity_checks"] = _identity_checks(report, source)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rolling-origin validation.")
    ap.add_argument(
        "--verify", action="store_true",
        help="regenerate and require the committed artifact byte for byte; writes nothing",
    )
    args = ap.parse_args(argv)

    report = build_report()
    source = report["data_source"]
    n_folds = len(report["folds"])
    for f in report["folds"]:
        t05, t95 = f["thresholds"]
        print(
            f"[fold {f['fold']}] train={f['train']['n_rows']} test={f['test']['n_rows']} "
            f"({f['test']['n_fraud']} fraud) PR-AUC {f['pr_auc']['value']} "
            f"[{f['pr_auc']['ci_lower']}, {f['pr_auc']['ci_upper']}] "
            f"recall@0.5 {t05['recall']['value']} recall@0.95 {t95['recall']['value']} "
            f"ECE {f['calibration']['ece_platt']['value']}"
        )
    p = report["pooled"]
    print(f"[pool ] {p['n_fraud']} frauds PR-AUC {p['pr_auc']['value']} "
          f"[{p['pr_auc']['ci_lower']}, {p['pr_auc']['ci_upper']}]")
    if n_folds < MIN_FOLDS:
        print(f"FAIL: {n_folds} folds after merging short windows; need {MIN_FOLDS}. "
              "Nothing written.")
        return 1

    payload = json.dumps(report, indent=2) + "\n"
    rel = f"artifacts/rolling_origin_{source}.json"
    if args.verify:
        try:
            committed = subprocess.run(
                ["git", "show", f"HEAD:{rel}"], cwd=REPO_ROOT, capture_output=True, check=True,
            ).stdout.decode()
        except (OSError, subprocess.CalledProcessError):
            print(f"FAIL: {rel} is not committed")
            return 1
        if committed.replace("\r\n", "\n") != payload:
            print(f"FAIL: regenerated report differs from the committed {rel}")
            return 1
        print(f"PASS: regenerated report is byte-identical to the committed {rel}")
        return 0

    out = get_settings().artifact_dir / f"rolling_origin_{source}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload)
    print(f"[save ] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
