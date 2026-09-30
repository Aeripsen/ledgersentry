"""
Train + evaluate LedgerSentry's baseline fraud detector with a leakage-safe
split, then write the serving artifact and a metrics report.

Run:  python scripts/train.py     (or, with the package installed:  python -m ledgersentry.train)

Prints the real measured PR-AUC and the coverage-vs-precision reject-knob table.
Only real, measured numbers are ever written here or to artifacts/metrics.json -
no invented "$ saved" or impact figures. See docs/model_card.md.
"""
from __future__ import annotations

import argparse
import json

import joblib
from sklearn.metrics import average_precision_score, roc_auc_score

from . import tracking
from .config import get_settings
from .data import (
    build_preprocessor,
    engineer_time_features,
    feature_columns,
    load,
    temporal_grouped_split,
)
from .drift import reference_stats
from .model import FraudDetector


def main(model_name: str | None = None) -> dict:
    cfg = get_settings()
    model_name = model_name or cfg.model

    df = load(data_dir=cfg.data_dir)
    source = df.attrs.get("source", "unknown")
    print(f"[data ] source={source} rows={len(df)} fraud_rate={df['is_fraud'].mean():.4%}")

    df = engineer_time_features(df)
    train_df, test_df = temporal_grouped_split(df, test_size=cfg.test_size)
    print(
        f"[split] train={len(train_df)} test={len(test_df)} "
        f"(grouped by entity_id + time-ordered cohorts, no entity in both)"
    )

    pre = build_preprocessor(train_df)
    X_train = pre.fit_transform(train_df)  # fit on TRAIN only
    X_test = pre.transform(test_df)
    y_train = train_df["is_fraud"].to_numpy()
    y_test = test_df["is_fraud"].to_numpy()

    print(f"[fit  ] FraudDetector (model={model_name}, balanced sample weights) ...")
    model = FraudDetector(
        random_state=cfg.random_state,
        max_iter=cfg.max_iter,
        learning_rate=cfg.learning_rate,
        model=model_name,
    )
    model.fit(X_train, y_train)

    p_fraud = model.predict_proba_fraud(X_test)
    pr_auc = float(average_precision_score(y_test, p_fraud))
    # PR-AUC of a random/no-skill scorer equals the positive (fraud) rate.
    random_baseline = float(y_test.mean())
    curve = model.coverage_precision_curve(X_test, y_test, cfg.review_thresholds)

    # The two metrics ADR 002 rejected, computed on these exact predictions so the
    # ADR's argument is demonstrated instead of asserted. They are EVIDENCE, never
    # the headline - see the "demoted_metrics.why" string written into the artifact.
    roc_auc = float(roc_auc_score(y_test, p_fraud))
    accuracy = float(((p_fraud >= 0.5).astype(int) == y_test).mean())
    # the do-nothing baseline: predict "legit" for every row, catch zero fraud
    accuracy_always_legit = float((y_test == 0).mean())

    # Headline recall a fraud desk asks for first: at full automation (threshold
    # 0.5, nothing sent to review) what fraction of real fraud does the automated
    # path catch. curve[0] is the 0.5 row (REVIEW_THRESHOLDS[0]).
    n_test_fraud = int(y_test.sum())
    recall_full = curve[0]["recall_auto"]

    metrics = {
        "data_source": source,
        "is_synthetic": source == "synthetic",
        "model": model_name,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "train_fraud_rate": round(float(y_train.mean()), 4),
        "test_fraud_rate": round(float(y_test.mean()), 4),
        "n_test_fraud": n_test_fraud,
        "pr_auc": round(pr_auc, 4),
        "pr_auc_random_baseline": round(random_baseline, 4),
        "recall_at_full_coverage": recall_full,
        # Deliberately nested under a self-describing key rather than sitting
        # flat beside pr_auc: these numbers exist to be quoted WITH their caveat,
        # and a flat "roc_auc" would eventually be lifted out as a headline by
        # someone skimming. The caveat travels with the number.
        "demoted_metrics": {
            "why": (
                "ADR 002 rejected these two as headlines at a 0.13% base rate and "
                "they are reported here as evidence for that decision, measured on "
                "the same predictions as the PR-AUC above. Never quote them as the "
                "headline. accuracy is worse than accuracy_always_predict_legit, "
                "which catches zero fraud: that is the argument, not a defect."
            ),
            "roc_auc": round(roc_auc, 4),
            # ROC-AUC's no-skill baseline is 0.5 at ANY imbalance, which is exactly
            # why it flatters here; PR-AUC's moves with the fold (pr_auc_random_
            # baseline above). The pair is the whole point.
            "roc_auc_no_skill_baseline": 0.5,
            "accuracy": round(accuracy, 4),
            "accuracy_always_predict_legit": round(accuracy_always_legit, 4),
        },
        "coverage_precision_curve": curve,
    }

    label = "SYNTHETIC FIXTURE" if metrics["is_synthetic"] else f"REAL DATA ({source})"
    caught = curve[0]["fraud_caught_auto"]
    print(f"[eval ] {label}: PR-AUC={pr_auc:.4f} (no-skill baseline {random_baseline:.4f})")
    print(
        f"[recall] full-automation recall {recall_full:.4f} "
        f"({caught}/{n_test_fraud} test frauds auto-caught)"
    )
    # printed so the ADR 002 argument is visible in the run itself, not just the file
    print(
        f"[demoted] ROC-AUC={roc_auc:.4f} on these same predictions (no-skill 0.5 at any "
        f"imbalance); accuracy={accuracy:.4f} vs {accuracy_always_legit:.4f} for "
        f"always-predict-legit, which catches 0 fraud. Evidence for ADR 002, never the headline."
    )

    # drift reference: the TRAIN split's per-feature distribution, frozen into
    # the artifact beside the model it describes (see drift.py). Numeric
    # features only; the categorical column is one-hot and low-cardinality.
    numeric_cols, _ = feature_columns(train_df)
    drift_reference = reference_stats(train_df, numeric_cols, bins=cfg.psi_bins)

    artifact_dir = cfg.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"preprocessor": pre, "model": model, "drift_reference": drift_reference},
        artifact_dir / "ledgersentry.joblib",
    )
    payload = json.dumps(metrics, indent=2)
    # metrics.json is always the latest run; metrics_<source>.json is a per-source
    # snapshot so a real-data run and the synthetic CI fixture can sit side by side
    # in git without one silently overwriting the other's numbers.
    (artifact_dir / "metrics.json").write_text(payload)
    (artifact_dir / f"metrics_{source}.json").write_text(payload)

    print(json.dumps(metrics, indent=2))
    print(f"[save ] {artifact_dir / 'ledgersentry.joblib'}")

    # Experiment tracking, after both metrics files are written so it can never
    # change the bytes verify_repro.py checks. A no-op unless mlflow is installed
    # and LEDGERSENTRY_MLFLOW is not 0; see tracking.py.
    if tracking.enabled():
        run_id, same = tracking.log_training_run(
            settings=cfg,
            metrics=metrics,
            preprocessor=pre,
            model=model,
            X_test=test_df,
            p_fraud_test=p_fraud,
            metrics_path=artifact_dir / f"metrics_{source}.json",
        )
        print(f"[mlflow] run {run_id} -> {tracking.tracking_uri()} "
              f"(logged model reproduces P(fraud) on the test fold: {same})")
    return metrics


def cli() -> None:
    ap = argparse.ArgumentParser(description="Train + evaluate LedgerSentry.")
    ap.add_argument(
        "--model", default=None,
        help="registry model name (default: config / hist_gbdt); see registry.py",
    )
    args = ap.parse_args()
    main(model_name=args.model)


if __name__ == "__main__":
    cli()
