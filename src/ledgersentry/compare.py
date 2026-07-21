"""
Does the velocity family help, and is the default boosting config the right one?

Both questions were open: train.py has always fit one model with one feature set
and reported it, so "hist_gbdt at 200 rounds and lr 0.1 on hour_of_day +
day_of_week + V1..V28" was a starting guess that became the committed headline
by never being challenged.

This pipeline runs every (feature set x boosting config) pair through the SAME
strictly temporal holdout train.py uses, so the numbers sit next to the
committed 0.7278 and mean the same thing.

The rules it holds itself to:

  Selection is blind to the test fold. Each variant is first fit on an inner
  train slice and scored on an inner validation slice, both carved out of the
  TRAIN window with the same leakage-safe split function. The winner is chosen
  on validation PR-AUC alone. The test fold is scored for every variant,
  winners and losers, and printed - but it never chooses anything. Picking the
  best test number out of eight and calling it the result is how a holdout
  quietly becomes a training set.

  Every variant is reported. A comparison that only publishes its winner is an
  advertisement.

  "Better" gets an interval. The headline fold holds 75 frauds, so a PR-AUC gap
  of a few points is inside the noise. The delta between the incumbent and the
  challenger is bootstrapped PAIRED (both models scored on the same resampled
  rows), which cancels the fold-draw variance the two share and is the only
  version of the comparison that can distinguish "helped" from "moved".

Run: python scripts/compare.py   -> artifacts/comparison_<source>.json
"""
from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score

from .bootstrap import bootstrap_headline
from .config import get_settings
from .data import build_preprocessor, engineer_time_features, load, temporal_grouped_split
from .model import FraudDetector
from .velocity import add_velocity_features, degenerate_columns, velocity_columns


@dataclass(frozen=True)
class GbdtConfig:
    name: str
    model: str
    max_iter: int
    learning_rate: float


# The first entry is the incumbent: exactly the config behind the committed
# metrics. Everything else is measured against it.
CONFIGS: tuple[GbdtConfig, ...] = (
    GbdtConfig("gbdt_default", "hist_gbdt", 200, 0.1),
    GbdtConfig("gbdt_slow", "hist_gbdt", 400, 0.05),
    GbdtConfig("gbdt_shallow", "hist_gbdt_shallow", 400, 0.05),
    GbdtConfig("gbdt_deep", "hist_gbdt_deep", 200, 0.1),
)

FEATURE_SETS = ("base", "base+velocity")


def _fit_score(
    train_df: pd.DataFrame, eval_df: pd.DataFrame, cfg_obj: GbdtConfig, random_state: int
) -> tuple[np.ndarray, float]:
    pre = build_preprocessor(train_df)
    X_train = pre.fit_transform(train_df)
    X_eval = pre.transform(eval_df)
    started = time.perf_counter()
    model = FraudDetector(
        random_state=random_state,
        max_iter=cfg_obj.max_iter,
        learning_rate=cfg_obj.learning_rate,
        model=cfg_obj.model,
    ).fit(X_train, train_df["is_fraud"].to_numpy())
    fit_seconds = time.perf_counter() - started
    return model.predict_proba_fraud(X_eval), fit_seconds


def _drop_dead_columns(frame: pd.DataFrame, test_size: float) -> list[str]:
    """Velocity columns that never vary in training. Judged on the train window
    and on the inner train slice (both are fit on, so a column constant in
    either one is dead weight there), never on the test fold."""
    train_df, _ = temporal_grouped_split(frame, test_size=test_size)
    inner_train, _ = temporal_grouped_split(train_df, test_size=test_size)
    cols = velocity_columns(frame)
    dead = set(degenerate_columns(train_df, cols)) | set(
        degenerate_columns(inner_train, cols)
    )
    return sorted(dead)


def _recall_at_half(y: np.ndarray, p: np.ndarray) -> tuple[int, float | None]:
    total = int((y == 1).sum())
    caught = int(((p >= 0.5) & (y == 1)).sum())
    return caught, (caught / total if total else None)


def paired_delta_bootstrap(
    y: np.ndarray,
    p_incumbent: np.ndarray,
    p_challenger: np.ndarray,
    n_resamples: int,
    seed: int,
) -> dict:
    """95% interval on (challenger PR-AUC - incumbent PR-AUC), resampling the
    test fold once per iteration and scoring BOTH models on that same resample.

    Unpaired intervals on two PR-AUCs from one fold overlap almost always at 75
    positives, which reads as "no difference" even when one model is better on
    every single draw. Pairing removes the shared draw and leaves the part that
    is actually about the models."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        if int((y_b == 1).sum()) == 0:
            continue
        deltas.append(
            float(average_precision_score(y_b, p_challenger[idx]))
            - float(average_precision_score(y_b, p_incumbent[idx]))
        )
    arr = np.asarray(deltas)
    lo, hi = (float(v) for v in np.percentile(arr, (2.5, 97.5)))
    point = float(average_precision_score(y, p_challenger)) - float(
        average_precision_score(y, p_incumbent)
    )
    return {
        "delta_pr_auc": round(point, 4),
        "ci_lower": round(lo, 4),
        "ci_upper": round(hi, 4),
        "share_of_resamples_challenger_wins": round(float((arr > 0).mean()), 4),
        "interval_excludes_zero": bool(lo > 0 or hi < 0),
    }


def main() -> dict:
    cfg = get_settings()

    df = load(data_dir=cfg.data_dir)
    source = df.attrs.get("source", "unknown")
    df = engineer_time_features(df)

    velocity_frame = add_velocity_features(df)
    dropped = _drop_dead_columns(velocity_frame, cfg.test_size)
    velocity_frame = velocity_frame.drop(columns=dropped)
    frames = {"base": df, "base+velocity": velocity_frame}
    n_velocity = len(velocity_columns(velocity_frame))
    print(
        f"[data ] source={source} rows={len(df)} velocity_columns={n_velocity} "
        f"(dropped {len(dropped)} with no variation in training)"
    )

    splits = {}
    for name, frame in frames.items():
        train_df, test_df = temporal_grouped_split(frame, test_size=cfg.test_size)
        # The inner cut reuses the same function on the train window only, so
        # validation is later in time than what the model saw, exactly like the
        # real holdout is.
        inner_train, inner_val = temporal_grouped_split(train_df, test_size=cfg.test_size)
        splits[name] = (train_df, test_df, inner_train, inner_val)

    train_df, test_df, inner_train, inner_val = splits["base"]
    # The split keys off entity_id and timestamp only, so adding feature columns
    # cannot move a row across the boundary. Checked rather than assumed: if it
    # ever did, the two feature sets would be graded on different folds and the
    # whole comparison would be meaningless.
    for name, (tr, te, itr, iva) in splits.items():
        assert (len(tr), len(te), len(itr), len(iva)) == (
            len(train_df), len(test_df), len(inner_train), len(inner_val)
        ), f"{name} split differs from the base split"

    y_test = test_df["is_fraud"].to_numpy()
    y_val = inner_val["is_fraud"].to_numpy()
    print(
        f"[split] train={len(train_df)} ({int(train_df['is_fraud'].sum())} fraud) "
        f"test={len(test_df)} ({int(y_test.sum())} fraud) | inner train={len(inner_train)} "
        f"val={len(inner_val)} ({int(y_val.sum())} fraud)"
    )

    results: list[dict] = []
    test_scores: dict[str, np.ndarray] = {}
    for fs in FEATURE_SETS:
        tr, te, itr, iva = splits[fs]
        for c in CONFIGS:
            key = f"{fs}|{c.name}"
            p_val, _ = _fit_score(itr, iva, c, cfg.random_state)
            p_test, fit_seconds = _fit_score(tr, te, c, cfg.random_state)
            test_scores[key] = p_test
            caught, recall = _recall_at_half(y_test, p_test)
            row = {
                "feature_set": fs,
                "config": c.name,
                "model": c.model,
                "max_iter": c.max_iter,
                "learning_rate": c.learning_rate,
                "val_pr_auc": round(float(average_precision_score(y_val, p_val)), 4),
                "test_pr_auc": round(float(average_precision_score(y_test, p_test)), 4),
                "test_recall_at_full_coverage": round(recall, 4) if recall is not None else None,
                "test_fraud_caught": caught,
                "fit_seconds": round(fit_seconds, 1),
            }
            results.append(row)
            print(
                f"[run  ] {key:34s} val PR-AUC {row['val_pr_auc']:.4f} "
                f"test PR-AUC {row['test_pr_auc']:.4f} "
                f"recall {row['test_recall_at_full_coverage']} ({caught}/{int(y_test.sum())})"
            )

    incumbent = next(
        r for r in results if r["feature_set"] == "base" and r["config"] == "gbdt_default"
    )
    selected = max(results, key=lambda r: r["val_pr_auc"])
    best_velocity = max(
        (r for r in results if r["feature_set"] == "base+velocity"),
        key=lambda r: r["val_pr_auc"],
    )
    print(
        f"[pick ] selected on validation only: {selected['feature_set']}|{selected['config']} "
        f"(val {selected['val_pr_auc']:.4f}) -> test PR-AUC {selected['test_pr_auc']:.4f}"
    )

    # Priced deliberately, and labeled for what it is: the variant with the best
    # TEST number, which validation did not choose. Publishing its interval is
    # the point. Someone will find this row and want to quote it, and the
    # interval is the answer to why it is not the headline.
    best_on_test = max(results, key=lambda r: r["test_pr_auc"])

    p_incumbent = test_scores["base|gbdt_default"]
    deltas: dict[str, dict] = {}
    for label, row in (
        ("selected", selected),
        ("best_velocity", best_velocity),
        ("post_hoc_best_on_test", best_on_test),
    ):
        key = f"{row['feature_set']}|{row['config']}"
        if key == "base|gbdt_default":
            continue
        already = next((lbl for lbl, d in deltas.items() if d["variant"] == key), None)
        if already:
            deltas[label] = {"variant": key, "same_as": already}
            continue
        deltas[label] = {
            "variant": key,
            **paired_delta_bootstrap(
                y_test, p_incumbent, test_scores[key], cfg.bootstrap_resamples, cfg.random_state
            ),
        }
        d = deltas[label]
        print(
            f"[delta] {key} vs incumbent: {d['delta_pr_auc']:+.4f} PR-AUC, "
            f"95% CI [{d['ci_lower']}, {d['ci_upper']}], "
            f"wins {d['share_of_resamples_challenger_wins']:.0%} of resamples"
        )

    # Recompute the incumbent's own interval here as well: if this pipeline's
    # rebuild of the committed model does not land on the committed CI, the
    # comparison is describing some other model.
    incumbent_ci = bootstrap_headline(
        y_test, p_incumbent, n_resamples=cfg.bootstrap_resamples, seed=cfg.random_state
    )

    report = {
        "data_source": source,
        "is_synthetic": source == "synthetic",
        "measured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "split": {
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_test_fraud": int(y_test.sum()),
            "n_inner_train": int(len(inner_train)),
            "n_inner_val": int(len(inner_val)),
            "n_inner_val_fraud": int(y_val.sum()),
            "note": (
                "Identical to train.py's split (same function, same test_size, same "
                "frame order), so test_pr_auc here is comparable row for row with "
                "the committed metrics_<source>.json."
            ),
        },
        "n_velocity_columns": n_velocity,
        "velocity_columns_dropped_no_variation": dropped,
        "selection_rule": (
            "highest val_pr_auc on the inner validation slice, which is carved out "
            "of the train window and is later in time than the inner train slice. "
            "The test fold is reported for every variant and chooses nothing."
        ),
        "incumbent": f"{incumbent['feature_set']}|{incumbent['config']}",
        "selected_by_validation": f"{selected['feature_set']}|{selected['config']}",
        "best_on_test_not_selected": (
            None
            if best_on_test is selected
            else f"{best_on_test['feature_set']}|{best_on_test['config']}"
        ),
        "results": results,
        "paired_deltas_vs_incumbent": deltas,
        "incumbent_bootstrap": incumbent_ci,
        "how_to_read_this": [
            "test_pr_auc is reported for every variant including the ones "
            "validation rejected. Only selected_by_validation was chosen, and it "
            "was chosen without looking at any test_pr_auc in this file.",
            "best_on_test_not_selected is the variant with the best test number. "
            "It is here on purpose. If it were promoted to the headline, the "
            "holdout would have been used eight times to pick one model and the "
            "0.7278 would stop meaning what it says.",
            "Read the paired deltas, not the difference of two rounded PR-AUCs. "
            "On 75 positives a gap of a couple of points is inside the fold's "
            "own noise unless the paired interval says otherwise.",
        ],
    }
    if report["is_synthetic"]:
        report["synthetic_warning"] = (
            "The synthetic fixture builds its own f_entity_daily_tx_count by "
            "adding a random bump to fraud rows. The entity velocity features "
            "recompute that count honestly, so the pair of columns exposes the "
            "bump and any model reaches PR-AUC 1.0. That is a defect of the "
            "fixture's generator, not a result. Only the real-data file means "
            "anything here."
        )

    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.artifact_dir / f"comparison_{source}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[save ] {out}")
    return report


if __name__ == "__main__":
    main()
