#!/usr/bin/env python
"""Evidently data-drift + model-performance report over two TIME windows.
Run:  python scripts/evidently_report.py   (needs pip install -r requirements-mlops.txt)

Writes, per data source:
  reports/evidently_<source>.html          the Evidently report
  artifacts/evidently_summary_<source>.json the numbers in it, the repo's own PSI on
                                            the same windows, and the answer to "did
                                            drift come with a performance drop?"

## The windows

Both come from the split every committed number uses (temporal_grouped_split,
time-ordered), and both are scored by a model that never trained on them:

  reference  the inner validation slice: the LATEST 20% of the training window,
             scored by a model fit on the earlier 80% of it. This is the carve
             compare.py selects models on, so its PR-AUC is asserted equal to the
             incumbent's val_pr_auc in comparison_<source>.json.
  current    the held-out test fold (the last 20% of the data in time), scored by
             the shipped model fit on the whole training window. Its PR-AUC is
             asserted equal to metrics_<source>.json.

On ULB that is roughly day-2 afternoon (reference) against the final hours of
day 2 (current). The two scoring models differ in training size (80% vs 100% of
the training window); the report says so rather than hiding it. It also means a
PR-AUC change between the windows mixes a model difference with a window
difference, so it cannot be read as the effect of drift alone.

## Refusing to write

The committed metrics_<source>.json, comparison_<source>.json and
bootstrap_<source>.json must all exist, and both windows must land exactly on
them (current PR-AUC and its bootstrap CI, reference PR-AUC). A missing file is
a failure, not a skipped check: run train.py, compare.py and bootstrap.py first.

## Stattests, and why they differ from drift.py

Evidently's defaults at these sizes (> 1,000 rows): numeric columns get the
normed Wasserstein distance, drift if > 0.1; low-cardinality columns
(day_of_week here) are treated as categorical and get Jensen-Shannon distance,
drift if > 0.1. drift.py uses PSI on the reference window's deciles with the
config's 0.10 watch / 0.25 alert bands. Both run on the same two windows and
their agreement is counted, not assumed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ledgersentry.bootstrap import bootstrap_headline  # noqa: E402
from ledgersentry.config import get_settings  # noqa: E402
from ledgersentry.data import (  # noqa: E402
    build_preprocessor,
    engineer_time_features,
    feature_columns,
    load,
    temporal_grouped_split,
)
from ledgersentry.drift import drift_report, reference_stats  # noqa: E402
from ledgersentry.model import FraudDetector  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def fit_score(cfg: Any, fit_df: pd.DataFrame, score_df: pd.DataFrame) -> np.ndarray:
    pre = build_preprocessor(fit_df)
    X_fit = pre.fit_transform(fit_df)
    model = FraudDetector(random_state=cfg.random_state, max_iter=cfg.max_iter,
                          learning_rate=cfg.learning_rate, model=cfg.model)
    model.fit(X_fit, fit_df["is_fraud"].to_numpy())
    return model.predict_proba_fraud(pre.transform(score_df))


def classification_numbers(snapshot: Any) -> dict[str, float]:
    wanted = {"Accuracy": "accuracy", "Precision": "precision", "Recall": "recall",
              "F1Score": "f1", "RocAuc": "roc_auc", "LogLoss": "log_loss",
              "TPR": "tpr", "FPR": "fpr", "FNR": "fnr"}
    out: dict[str, float] = {}
    for m in snapshot.dict()["metrics"]:
        kind = m["config"]["type"].rsplit(":", 1)[-1]
        if kind in wanted and isinstance(m["value"], int | float):
            out[wanted[kind]] = round(float(m["value"]), 4)
    return out


def window_span(df: pd.DataFrame) -> dict[str, Any]:
    t = df["timestamp"]
    return {"rows": int(len(df)), "fraud": int(df["is_fraud"].sum()),
            "from": str(t.min()), "to": str(t.max()),
            "hours": round((t.max() - t.min()).total_seconds() / 3600, 2)}


def evidently_dataset_drift(cur_ds: Any, ref_ds: Any, columns: list[str]) -> dict[str, Any]:
    """Evidently's own dataset-level verdict: the default test DriftedColumnsCount
    attaches (share of drifted columns must stay below drift_share). Read from the
    test result, not recomputed here."""
    from evidently import Report
    from evidently.metrics import DriftedColumnsCount

    snap = Report([DriftedColumnsCount(columns=columns)], include_tests=True).run(
        current_data=cur_ds, reference_data=ref_ds)
    d = snap.dict()
    (test,) = [t for t in d["tests"]
               if t["metric_config"]["params"]["type"].endswith(":DriftedColumnsCount")]
    status = str(getattr(test["status"], "value", test["status"]))
    return {"dataset_drift": status == "FAIL", "evidently_test": test["name"],
            "evidently_test_status": status,
            "drift_share": float(test["metric_config"]["params"]["drift_share"]),
            "share": float(d["metrics"][0]["value"]["share"])}


def ci95(y: np.ndarray, p: np.ndarray, cfg: Any) -> list[float]:
    """The repo's percentile bootstrap (bootstrap.py), same resamples and seed."""
    b = bootstrap_headline(y, p, n_resamples=cfg.bootstrap_resamples, seed=cfg.random_state)
    return [b["pr_auc"]["ci_lower"], b["pr_auc"]["ci_upper"]]


def main() -> int:
    from evidently import BinaryClassification, DataDefinition, Dataset, Report
    from evidently.presets import ClassificationPreset, DataDriftPreset

    cfg = get_settings()
    df = load(data_dir=cfg.data_dir)
    source = str(df.attrs.get("source", "unknown"))
    df = engineer_time_features(df)
    train_df, test_df = temporal_grouped_split(df, test_size=cfg.test_size)
    inner_train, inner_val = temporal_grouped_split(train_df, test_size=cfg.test_size)
    numeric, _ = feature_columns(train_df)
    print(f"[data ] source={source} | reference (inner val) {len(inner_val)} rows, "
          f"current (test) {len(test_df)} rows | {len(numeric)} numeric features")

    p_ref = fit_score(cfg, inner_train, inner_val)
    p_cur = fit_score(cfg, train_df, test_df)
    y_ref = inner_val["is_fraud"].to_numpy()
    y_cur = test_df["is_fraud"].to_numpy()
    prauc_ref = round(float(average_precision_score(y_ref, p_ref)), 4)
    prauc_cur = round(float(average_precision_score(y_cur, p_cur)), 4)

    ci_cur, ci_ref = ci95(y_cur, p_cur, cfg), ci95(y_ref, p_ref, cfg)

    # both windows must be the numbers the repo already publishes; every committed
    # file is required, so a missing one fails instead of skipping its check
    need = {k: cfg.artifact_dir / f"{k}_{source}.json"
            for k in ("metrics", "comparison", "bootstrap")}
    missing = [f.name for f in need.values() if not f.exists()]
    if missing:
        print(f"[check] FAIL: committed file(s) missing: {missing}. Run train.py, "
              "compare.py and bootstrap.py first; nothing is written without them.")
        return 1
    comp = json.loads(need["comparison"].read_text())
    inc = next(r for r in comp["results"]
               if f"{r['feature_set']}|{r['config']}" == comp["incumbent"])
    boot = json.loads(need["bootstrap"].read_text())["pr_auc"]
    checks: dict[str, bool] = {
        "current_pr_auc_equals_metrics_json":
            prauc_cur == json.loads(need["metrics"].read_text())["pr_auc"],
        "current_ci_equals_bootstrap_json": ci_cur == [boot["ci_lower"], boot["ci_upper"]],
        "reference_pr_auc_equals_comparison_val": prauc_ref == inc["val_pr_auc"],
    }
    for k, ok in checks.items():
        print(f"[check] {k}: {ok}")
    if not all(checks.values()):
        print("[check] FAIL: a window does not reproduce the committed number")
        return 1

    cols = [*numeric, "is_fraud", "p_fraud"]
    ref_df = inner_val.assign(p_fraud=p_ref)[cols].reset_index(drop=True)
    cur_df = test_df.assign(p_fraud=p_cur)[cols].reset_index(drop=True)
    definition = DataDefinition(
        numerical_columns=numeric,
        classification=[BinaryClassification(target="is_fraud", prediction_probas="p_fraud",
                                             pos_label=1)],
    )
    ref_ds = Dataset.from_pandas(ref_df, data_definition=definition)
    cur_ds = Dataset.from_pandas(cur_df, data_definition=definition)

    print("[evidently] data drift + classification quality (threshold 0.5) ...")
    snapshot = Report(
        [DataDriftPreset(columns=numeric), ClassificationPreset()],
        metadata={"source": source,
                  "reference": "inner validation slice, model fit on inner train",
                  "current": "test fold, shipped model"},
    ).run(current_data=cur_ds, reference_data=ref_ds)
    html = ROOT / "reports" / f"evidently_{source}.html"
    html.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html))
    verdict = evidently_dataset_drift(cur_ds, ref_ds, numeric)
    ref_only = Report([ClassificationPreset()]).run(current_data=ref_ds)

    per_column: dict[str, dict[str, Any]] = {}
    drifted = share = None
    for m in snapshot.dict()["metrics"]:
        kind = m["config"]["type"].rsplit(":", 1)[-1]
        if kind == "DriftedColumnsCount":
            drifted, share = m["value"]["count"], m["value"]["share"]
        elif kind == "ValueDrift":
            c = m["config"]
            per_column[c["column"]] = {"method": c["method"], "threshold": c["threshold"],
                                       "score": round(float(m["value"]), 4),
                                       "drifted": float(m["value"]) > float(c["threshold"])}

    ref_stats = reference_stats(inner_val, numeric, bins=cfg.psi_bins)
    psi_rep = drift_report(ref_stats, test_df, cfg.psi_watch, cfg.psi_alert)
    psi_status = {f: v.get("status") for f, v in psi_rep["features"].items()}
    ev_flag = {f for f, r in per_column.items() if r["drifted"]}
    psi_flag = {f for f, s in psi_status.items() if s in ("watch", "alert")}
    methods: dict[str, int] = {}
    for r in per_column.values():
        methods[r["method"]] = methods.get(r["method"], 0) + 1

    if share is None or round(float(share), 6) != round(verdict["share"], 6):
        print("[check] FAIL: the dataset-drift test saw a different share than the report")
        return 1

    summary = {
        "what": ("Evidently data drift + classification quality over two time windows "
                 "of the committed split; both windows scored out of sample"),
        "data_source": source,
        "is_synthetic": source == "synthetic",
        "html_report": html.relative_to(ROOT).as_posix(),
        "evidently_version": __import__("evidently").__version__,
        "reference_window": {**window_span(inner_val),
                             "scored_by": "model fit on the inner train slice"},
        "current_window": {**window_span(test_df),
                           "scored_by": "shipped model, fit on the whole training window"},
        "checks_against_committed_artifacts": checks,
        "data_drift": {
            "drifted_columns": int(drifted) if drifted is not None else None,
            "n_columns": len(numeric),
            "drifted_share": round(float(share), 4) if share is not None else None,
            "dataset_drift": verdict["dataset_drift"],
            "dataset_drift_source": (
                f"Evidently's own test on DriftedColumnsCount: \"{verdict['evidently_test']}\" "
                f"-> {verdict['evidently_test_status']} (FAIL means dataset drift)"),
            "drift_share_threshold": verdict["drift_share"],
            "methods_used": dict(sorted(methods.items())),
            "drifted": {f: per_column[f] for f in sorted(ev_flag,
                                                        key=lambda f: -per_column[f]["score"])},
        },
        "repo_psi_same_windows": {
            "watch_or_alert": sorted(psi_flag),
            "alerts": psi_rep["n_alerts"],
            "worst_feature": psi_rep["worst_feature"],
            "worst_psi": psi_rep["worst_psi"],
            "bands": {"watch": cfg.psi_watch, "alert": cfg.psi_alert},
            "psi_of_evidently_flagged": {f: psi_rep["features"][f].get("psi")
                                         for f in sorted(ev_flag)},
            "agreement": {"both_flag": sorted(ev_flag & psi_flag),
                          "evidently_only": sorted(ev_flag - psi_flag),
                          "psi_only": sorted(psi_flag - ev_flag)},
        },
        "performance": {
            "pr_auc": {"reference": prauc_ref, "current": prauc_cur,
                       "change": round(prauc_cur - prauc_ref, 4),
                       "reference_bootstrap_ci95": ci_ref,
                       "current_bootstrap_ci95": ci_cur,
                       "bootstrap": (f"percentile, {cfg.bootstrap_resamples} resamples of "
                                     f"each window, seed {cfg.random_state} (bootstrap.py)"),
                       "reference_inside_current_ci": ci_cur[0] <= prauc_ref <= ci_cur[1],
                       "current_inside_reference_ci": ci_ref[0] <= prauc_cur <= ci_ref[1],
                       "caveat": ("the two windows are scored by different models (fit on "
                                  "80% vs 100% of the training window), so this change "
                                  "mixes a model difference with a window difference")},
            "at_threshold_0.5_reference": classification_numbers(ref_only),
            "at_threshold_0.5_current": classification_numbers(snapshot),
        },
    }
    out = cfg.artifact_dir / f"evidently_summary_{source}.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    d, p = summary["data_drift"], summary["performance"]["pr_auc"]
    print(f"[drift] Evidently {d['drifted_columns']}/{d['n_columns']} columns drifted "
          f"({sorted(ev_flag)}); PSI watch/alert: {sorted(psi_flag)}")
    print(f"[perf ] PR-AUC reference {p['reference']} (CI {ci_ref}) -> current "
          f"{p['current']} (CI {ci_cur})")
    print(f"[save ] {html}\n[save ] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
