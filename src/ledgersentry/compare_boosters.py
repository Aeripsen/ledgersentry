"""
Would LightGBM beat the sklearn HistGradientBoostingClassifier behind the
committed 0.7278?

The model card has argued since day one that hist_gbdt was chosen for
installability, not measured superiority ("xgboost is a documented drop-in
alternative"). That is an assertion in a repo whose whole claim is that claims
get measured. This pipeline measures it: the incumbent config against LightGBM
at the same boosting budget, on the exact holdout behind the headline.

It deliberately reuses compare.py's machinery instead of inventing its own:
the same _fit_score path (same preprocessor, same balanced weights), the same
temporal split, the same inner-validation selection rule, and the same paired
bootstrap on the delta (ADR 009). A booster comparison with its own slightly
different split would produce numbers that look comparable to the committed
ones and are not.

Scope decisions, stated up front:

  Base features only. The velocity family was measured on this fold and hurt
  (artifacts/comparison_ulb_creditcard.json), so the booster question is asked
  on the feature set the headline actually uses. Crossing boosters with a
  feature family already shown to hurt would double the fit count to decorate
  a settled question.

  Two LightGBM budgets, not a tuning sweep. lgbm_default mirrors the
  incumbent's budget exactly (200 rounds, lr 0.1, 31 leaves - LightGBM's own
  default leaf count, which happens to equal hist_gbdt's), so the delta is
  library vs library and nothing else. lgbm_slow (400 rounds, lr 0.05) mirrors
  the gbdt_slow variant already in the committed comparison. A real tuning
  sweep over num_leaves/min_child_samples would need nested validation this
  fold's 52 inner-validation frauds cannot support honestly.

Run: python scripts/compare_boosters.py -> artifacts/comparison_boosters_<source>.json
(needs the optional lightgbm install: make install-analysis)
"""
from __future__ import annotations

import json
import platform
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score

from . import tracking
from .bootstrap import bootstrap_headline
from .compare import GbdtConfig, _fit_score, paired_delta_bootstrap
from .config import get_settings
from .data import engineer_time_features, load, temporal_grouped_split

# First entry is the incumbent, exactly the config behind the committed
# metrics - same convention as compare.py, pinned by tests/test_compare_boosters.py.
CONFIGS: tuple[GbdtConfig, ...] = (
    GbdtConfig("gbdt_default", "hist_gbdt", 200, 0.1),
    GbdtConfig("lgbm_default", "lgbm", 200, 0.1),
    GbdtConfig("lgbm_slow", "lgbm", 400, 0.05),
)


def main() -> dict:
    cfg = get_settings()

    df = load(data_dir=cfg.data_dir)
    source = df.attrs.get("source", "unknown")
    df = engineer_time_features(df)
    print(f"[data ] source={source} rows={len(df)} (base features only, see docstring)")

    train_df, test_df = temporal_grouped_split(df, test_size=cfg.test_size)
    inner_train, inner_val = temporal_grouped_split(train_df, test_size=cfg.test_size)
    y_test = test_df["is_fraud"].to_numpy()
    y_val = inner_val["is_fraud"].to_numpy()
    print(
        f"[split] train={len(train_df)} test={len(test_df)} ({int(y_test.sum())} fraud) | "
        f"inner train={len(inner_train)} val={len(inner_val)} ({int(y_val.sum())} fraud)"
    )

    # Imported here, before the first slow fit, so a missing optional install
    # fails in seconds with the registry's clear error instead of after the
    # incumbent has already trained. Also supplies the version for the artifact.
    import lightgbm

    results: list[dict] = []
    test_scores: dict[str, np.ndarray] = {}
    for c in CONFIGS:
        p_val, _ = _fit_score(inner_train, inner_val, c, cfg.random_state)
        p_test, fit_seconds = _fit_score(train_df, test_df, c, cfg.random_state)
        test_scores[c.name] = p_test
        row = {
            "config": c.name,
            "model": c.model,
            "max_iter": c.max_iter,
            "learning_rate": c.learning_rate,
            "val_pr_auc": round(float(average_precision_score(y_val, p_val)), 4),
            "test_pr_auc": round(float(average_precision_score(y_test, p_test)), 4),
            "fit_seconds": round(fit_seconds, 1),
        }
        results.append(row)
        print(
            f"[run  ] {c.name:14s} val PR-AUC {row['val_pr_auc']:.4f} "
            f"test PR-AUC {row['test_pr_auc']:.4f} fit {row['fit_seconds']}s"
        )

    incumbent = results[0]
    selected = max(results, key=lambda r: r["val_pr_auc"])
    print(
        f"[pick ] selected on validation only: {selected['config']} "
        f"(val {selected['val_pr_auc']:.4f})"
    )

    # Every challenger gets a paired interval against the incumbent, winners
    # and losers alike - the same "a comparison that only publishes its winner
    # is an advertisement" rule compare.py holds itself to.
    p_incumbent = test_scores[CONFIGS[0].name]
    deltas: dict[str, dict] = {}
    for c in CONFIGS[1:]:
        deltas[c.name] = paired_delta_bootstrap(
            y_test, p_incumbent, test_scores[c.name],
            cfg.bootstrap_resamples, cfg.random_state,
        )
        d = deltas[c.name]
        print(
            f"[delta] {c.name} vs incumbent: {d['delta_pr_auc']:+.4f} PR-AUC, "
            f"95% CI [{d['ci_lower']}, {d['ci_upper']}], "
            f"wins {d['share_of_resamples_challenger_wins']:.0%} of resamples"
        )

    # Same identity check as compare.py: if this rebuild of the incumbent does
    # not land on the committed CI, everything above describes some other model.
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
            "lightgbm": lightgbm.__version__,
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
                "Identical to train.py's and compare.py's split, so every number "
                "here is comparable row for row with the committed "
                "metrics_<source>.json and comparison_<source>.json."
            ),
        },
        "feature_set": "base",
        "feature_set_note": (
            "Base features only: velocity was measured on this fold and hurt "
            "(comparison_<source>.json), so the booster question is asked on the "
            "feature set the headline uses."
        ),
        "selection_rule": (
            "highest val_pr_auc on the inner validation slice; the test fold is "
            "reported for every config and chooses nothing."
        ),
        "incumbent": incumbent["config"],
        "selected_by_validation": selected["config"],
        "results": results,
        "paired_deltas_vs_incumbent": deltas,
        "incumbent_bootstrap": incumbent_ci,
        "how_to_read_this": [
            "Read the paired deltas, not the difference of two rounded PR-AUCs: "
            "on this fold's positives a gap of a couple of points is inside the "
            "fold's own noise unless the paired interval says otherwise.",
            "interval_excludes_zero false means the honest conclusion is 'no "
            "measured difference between the libraries here', not 'LightGBM is "
            "worse'. A null between two competent boosters on one fold is the "
            "expected result, and this repo commits its nulls.",
        ],
    }

    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.artifact_dir / f"comparison_boosters_{source}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[save ] {out}")
    if tracking.enabled():
        readback = cfg.artifact_dir / f"mlflow_comparison_boosters_{source}.json"
        ids = tracking.log_comparison("boosters", report, data_dir=cfg.data_dir,
                                      readback_path=readback)
        print(f"[mlflow] {len(ids)} runs -> {tracking.tracking_uri()}; read back from "
              f"the store, every value equal to this report -> {readback.name}")
    return report


if __name__ == "__main__":
    main()
