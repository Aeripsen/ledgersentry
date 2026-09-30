"""
Experiment tracking seam (MLflow). Optional: serving, the base install and
`make reproduce` never import mlflow.

What gets logged, once mlflow is installed (pip install -r requirements-mlops.txt):
  * train.py: every training run, experiment `ledgersentry-train`. Params (the
    whole training config), the headline metrics, the demoted ones under a
    `demoted.` prefix, the review-knob curve as stepped metrics, the bootstrap CI
    when artifacts/bootstrap_<source>.json matches this run, metrics.json, and the
    fitted preprocessor+model saved with skops. The logged model is loaded back
    and must reproduce P(fraud) on every test row, or the run is tagged failed.
  * compare.py / compare_boosters.py: one run per variant as the script measures
    it, experiment `ledgersentry-compare`, with its validation and test PR-AUC,
    the paired-bootstrap delta against the incumbent, and which variant the
    validation rule selected.

Every run carries data_source and is_synthetic tags, because CI trains on the
synthetic fixture and a synthetic PR-AUC of 1.0 must never be read as a result.

Where runs go: a local SQLite store at <repo>/mlflow.db with artifacts under
<repo>/mlruns/ (both gitignored). View with
    mlflow ui --backend-store-uri sqlite:///mlflow.db
MLFLOW_TRACKING_URI overrides the store; LEDGERSENTRY_MLFLOW=0 turns it off.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO_ROOT / "mlflow.db"
DEFAULT_ARTIFACTS = REPO_ROOT / "mlruns"
SWITCH = "LEDGERSENTRY_MLFLOW"

# mlflow 3 saves sklearn models with skops, which refuses any type nobody vouched
# for. These are the two non-sklearn-core types in the shipped pipeline, each
# reviewed: our own FraudDetector wrapper, and the fitted tree predictor inside
# HistGradientBoostingClassifier. Everything else (ColumnTransformer, the
# encoders, the logreg baseline) is on skops' default trusted list.
SKOPS_TRUSTED_TYPES = [
    "ledgersentry.model.FraudDetector",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]


def enabled() -> bool:
    if os.environ.get(SWITCH, "1").strip() == "0":
        return False
    try:
        import mlflow  # noqa: F401
    except ImportError:
        return False
    return True


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{DEFAULT_DB.as_posix()}"


def use_experiment(name: str) -> str:
    import mlflow

    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    mlflow.set_tracking_uri(tracking_uri())
    exp = mlflow.get_experiment_by_name(name)
    if exp is not None:
        mlflow.set_experiment(experiment_id=exp.experiment_id)
        return str(exp.experiment_id)
    location = DEFAULT_ARTIFACTS.as_uri() if "MLFLOW_TRACKING_URI" not in os.environ else None
    exp_id = mlflow.create_experiment(name, artifact_location=location)
    mlflow.set_experiment(experiment_id=exp_id)
    return str(exp_id)


def metric_key(name: str) -> str:
    """mlflow metric names allow alphanumerics and _ - . / and space only."""
    return "".join(c if c.isalnum() or c in "_-./ " else "_" for c in name)


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_tags(source: str, data_dir: Path) -> dict[str, str]:
    tags = {"git_sha": git_sha(), "data_source": source,
            "is_synthetic": str(source == "synthetic")}
    if source == "ulb_creditcard" and (data_dir / "creditcard.csv").exists():
        tags["data.creditcard_csv_sha256"] = file_sha256(data_dir / "creditcard.csv")
    return tags


def pip_requirements() -> list[str]:
    from importlib.metadata import PackageNotFoundError, version

    reqs = []
    for n in ("scikit-learn", "numpy", "pandas", "scipy", "joblib"):
        try:
            reqs.append(f"{n}=={version(n)}")
        except PackageNotFoundError:
            continue
    return reqs


def _scalars(d: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    return {
        metric_key(f"{prefix}{k}"): float(v)
        for k, v in d.items()
        if isinstance(v, int | float) and not isinstance(v, bool)
    }


def log_training_run(
    *,
    settings: Any,
    metrics: Mapping[str, Any],
    preprocessor: Any,
    model: Any,
    X_test: Any,
    p_fraud_test: np.ndarray,
    metrics_path: Path,
) -> tuple[str, bool]:
    """Log one train.py run. Returns (run id, logged model reproduces P(fraud))."""
    import json

    import mlflow
    import mlflow.sklearn
    from sklearn.pipeline import Pipeline

    source = str(metrics["data_source"])
    use_experiment("ledgersentry-train")
    with mlflow.start_run(run_name=f"{metrics['model']}-{source}") as run:
        mlflow.set_tags({**data_tags(source, settings.data_dir),
                         "entrypoint": "ledgersentry.train"})
        mlflow.log_params({
            "model": metrics["model"],
            "random_state": settings.random_state,
            "max_iter": settings.max_iter,
            "learning_rate": settings.learning_rate,
            "test_size": settings.test_size,
            "split": "temporal_grouped_split (time-ordered, grouped by entity_id)",
            "sample_weight": "balanced, from train labels",
            "review_thresholds": ",".join(str(t) for t in settings.review_thresholds),
        })
        mlflow.log_metrics(_scalars(metrics))
        mlflow.log_metrics(_scalars(metrics.get("demoted_metrics", {}), "demoted."))
        for row in metrics.get("coverage_precision_curve", []):
            step = int(round(float(row["review_threshold"]) * 100))
            for k in ("coverage", "precision_on_flagged", "recall_auto", "n_sent_to_review",
                      "fraud_caught_auto", "fraud_missed"):
                if isinstance(row.get(k), int | float):
                    mlflow.log_metric(f"curve.{k}", float(row[k]), step=step)
        # the bootstrap CI belongs to this run only if it was computed on the same
        # predictions; a stale file (different point estimate) is not logged
        boot = settings.artifact_dir / f"bootstrap_{source}.json"
        if boot.exists():
            b = json.loads(boot.read_text())
            if b.get("pr_auc", {}).get("point_estimate") == metrics["pr_auc"]:
                mlflow.log_metrics({"pr_auc.ci95_lower": b["pr_auc"]["ci_lower"],
                                    "pr_auc.ci95_upper": b["pr_auc"]["ci_upper"]})
                mlflow.log_artifact(str(boot))
        mlflow.log_artifact(str(metrics_path))

        pipe = Pipeline([("pre", preprocessor), ("model", model)])
        try:
            info = mlflow.sklearn.log_model(
                pipe, name="model", pip_requirements=pip_requirements(),
                code_paths=[str(REPO_ROOT / "src" / "ledgersentry")],
                skops_trusted_types=SKOPS_TRUSTED_TYPES)
            mlflow.set_tag("model_serialization", "skops")
        except Exception as exc:  # e.g. lgbm: skops cannot walk a compiled booster
            info = mlflow.sklearn.log_model(
                pipe, name="model", serialization_format="cloudpickle",
                pip_requirements=pip_requirements(),
                code_paths=[str(REPO_ROOT / "src" / "ledgersentry")])
            mlflow.set_tag("model_serialization", f"cloudpickle ({type(exc).__name__})")
        # round trip: what the store holds must score the test fold identically
        loaded = mlflow.sklearn.load_model(info.model_uri)
        p_back = loaded[-1].predict_proba_fraud(loaded[:-1].transform(X_test))
        same = bool(np.array_equal(p_back, p_fraud_test))
        mlflow.set_tag("roundtrip_predictions_identical", str(same))
        return str(run.info.run_id), same


def log_comparison(experiment_suffix: str, report: Mapping[str, Any], *,
                   data_dir: Path) -> list[str]:
    """One run per variant of a compare.py / compare_boosters.py report, logged
    from the report the script just computed (not from a file on disk)."""
    import mlflow

    source = str(report["data_source"])
    use_experiment("ledgersentry-compare")
    deltas: dict[str, Mapping[str, Any]] = {}
    for key, d in report.get("paired_deltas_vs_incumbent", {}).items():
        deltas[str(d.get("variant", key))] = d
    tags = data_tags(source, data_dir)
    ids = []
    for row in report["results"]:
        variant = (f"{row['feature_set']}|{row['config']}" if "feature_set" in row
                   else str(row["config"]))
        with mlflow.start_run(run_name=f"{experiment_suffix}:{variant}") as run:
            mlflow.set_tags({
                **tags, "comparison": experiment_suffix, "variant": variant,
                "incumbent": str(report["incumbent"]),
                "is_incumbent": str(variant == report["incumbent"]),
                "selected_by_validation": str(variant == report["selected_by_validation"]),
                "selection_rule": str(report["selection_rule"])[:5000],
            })
            mlflow.log_params({k: v for k, v in row.items()
                               if k in ("feature_set", "config", "model", "max_iter",
                                        "learning_rate")})
            mlflow.log_params({f"split.{k}": v for k, v in report["split"].items()
                               if k != "note"})
            mlflow.log_metrics(_scalars({k: v for k, v in row.items()
                                         if k not in ("max_iter", "learning_rate")}))
            if variant in deltas:
                d = deltas[variant]
                mlflow.log_metrics(_scalars(
                    {k: v for k, v in d.items() if k != "interval_excludes_zero"},
                    "paired_delta_vs_incumbent."))
                mlflow.set_tag("paired_delta_excludes_zero",
                               str(d.get("interval_excludes_zero")))
            mlflow.log_dict(dict(report), f"{experiment_suffix}_report.json")
            ids.append(str(run.info.run_id))
    return ids
