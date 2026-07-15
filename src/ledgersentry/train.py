"""
Train + evaluate LedgerSentry's baseline fraud detector with a leakage-safe
split, then write the serving artifact and a metrics report.

Run:  python scripts/train.py     (or, with the package installed:  python -m ledgersentry.train)

Prints the real measured PR-AUC and the coverage-vs-precision reject-knob table.
Only real, measured numbers are ever written here or to artifacts/metrics.json -
no invented "$ saved" or impact figures. See docs/model_card.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.metrics import average_precision_score

from .data import build_preprocessor, engineer_time_features, load, temporal_grouped_split
from .model import FraudDetector

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
REVIEW_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]


def main() -> dict:
    df = load()
    source = df.attrs.get("source", "unknown")
    print(f"[data ] source={source} rows={len(df)} fraud_rate={df['is_fraud'].mean():.4%}")

    df = engineer_time_features(df)
    train_df, test_df = temporal_grouped_split(df, test_size=0.2)
    print(
        f"[split] train={len(train_df)} test={len(test_df)} "
        f"(grouped by entity_id + time-ordered cohorts, no entity in both)"
    )

    pre = build_preprocessor(train_df)
    X_train = pre.fit_transform(train_df)  # fit on TRAIN only
    X_test = pre.transform(test_df)
    y_train = train_df["is_fraud"].to_numpy()
    y_test = test_df["is_fraud"].to_numpy()

    print("[fit  ] FraudDetector (HistGradientBoostingClassifier, balanced sample weights) ...")
    model = FraudDetector()
    model.fit(X_train, y_train)

    p_fraud = model.predict_proba_fraud(X_test)
    pr_auc = float(average_precision_score(y_test, p_fraud))
    # PR-AUC of a random/no-skill scorer equals the positive (fraud) rate.
    random_baseline = float(y_test.mean())
    curve = model.coverage_precision_curve(X_test, y_test, REVIEW_THRESHOLDS)

    metrics = {
        "data_source": source,
        "is_synthetic": source == "synthetic",
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "train_fraud_rate": round(float(y_train.mean()), 4),
        "test_fraud_rate": round(float(y_test.mean()), 4),
        "pr_auc": round(pr_auc, 4),
        "pr_auc_random_baseline": round(random_baseline, 4),
        "coverage_precision_curve": curve,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"preprocessor": pre, "model": model}, ARTIFACT_DIR / "ledgersentry.joblib")
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    print(f"[save ] {ARTIFACT_DIR / 'ledgersentry.joblib'}")
    return metrics


if __name__ == "__main__":
    main()
