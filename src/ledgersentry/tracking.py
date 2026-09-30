"""
Experiment tracking seam (MLflow). Optional: serving, the base install and
`make reproduce` never import mlflow.

What gets logged, once mlflow is installed (pip install -r requirements-mlops.txt):
  * train.py: every training run, experiment `ledgersentry-train`. Params (the
    whole training config), the headline metrics, the demoted ones under a
    `demoted.` prefix, the review-knob curve as stepped metrics, the bootstrap CI
    when artifacts/bootstrap_<source>.json matches this run, metrics.json, and the
    fitted preprocessor+model (skops for sklearn-native models such as the default
    hist_gbdt; cloudpickle only for a LightGBM booster, which skops cannot walk).
    The logged model is loaded back and must reproduce P(fraud) on every test row
    exactly, or RoundTripError is raised and train.py exits non-zero. It is logged
    in MLflow's sklearn flavor only: FraudDetector has no predict(), so MLflow
    adds no python_function flavor and `mlflow models serve` does not apply.
  * compare.py / compare_boosters.py: one run per variant as the script measures
    it, experiment `ledgersentry-compare`, with its validation and test PR-AUC,
    which variant the validation rule selected, and, for the variants the report
    pairs against the incumbent, the paired-bootstrap delta. The runs are then
    read back from the store, every logged value is compared with the report the
    script computed (mismatch raises), and the read-back is written to
    artifacts/mlflow_<comparison>_<source>.json, which CI diffs.

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


class RoundTripError(RuntimeError):
    """The model read back from the tracking store scores differently from the
    model that was logged."""


class ReadBackError(RuntimeError):
    """A run read back from the tracking store disagrees with what was logged."""


def serialization_for(model: Any) -> str:
    """skops for sklearn-native estimators (the default hist_gbdt and the logreg
    baseline); cloudpickle only for a LightGBM booster, which skops cannot walk.
    Chosen by type up front, so a skops failure on a sklearn model is an error,
    never a silent fall back to pickle."""
    inner = getattr(model, "model_", model)
    final = inner.steps[-1][1] if hasattr(inner, "steps") else inner
    return "cloudpickle" if type(final).__module__.split(".")[0] == "lightgbm" else "skops"


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
        fmt = serialization_for(model)
        common = {"name": "model", "pip_requirements": pip_requirements(),
                  "code_paths": [str(REPO_ROOT / "src" / "ledgersentry")]}
        if fmt == "skops":
            info = mlflow.sklearn.log_model(pipe, skops_trusted_types=SKOPS_TRUSTED_TYPES,
                                            **common)
            mlflow.set_tag("model_serialization", "skops")
        else:
            info = mlflow.sklearn.log_model(pipe, serialization_format="cloudpickle", **common)
            mlflow.set_tag("model_serialization",
                           "cloudpickle (LightGBM booster; load only from a store you wrote)")
        # round trip: what the store holds must score the test fold identically
        loaded = mlflow.sklearn.load_model(info.model_uri)
        p_back = np.asarray(loaded[-1].predict_proba_fraud(loaded[:-1].transform(X_test)))
        same = bool(p_back.shape == np.shape(p_fraud_test)
                    and np.array_equal(p_back, p_fraud_test))
        mlflow.set_tag("roundtrip_predictions_identical", str(same))
        mlflow.log_metric("roundtrip.n_rows", len(p_back))
        if not same:
            raise RoundTripError(
                f"logged model {info.model_uri} does not reproduce P(fraud) on the "
                f"{len(p_back)} test rows")
        return str(run.info.run_id), same


def _variant(row: Mapping[str, Any]) -> str:
    return (f"{row['feature_set']}|{row['config']}" if "feature_set" in row
            else str(row["config"]))


# fit time is wall clock, so it is logged but kept out of the read-back file CI diffs
NOT_DETERMINISTIC = ("fit_seconds",)


def log_comparison(experiment_suffix: str, report: Mapping[str, Any], *,
                   data_dir: Path, readback_path: Path | None = None) -> list[str]:
    """One run per variant of a compare.py / compare_boosters.py report, logged
    from the report the script just computed (not from a file on disk). Then the
    runs are read back from the store and checked against the report; with
    readback_path set, the deterministic part of the read-back is written there."""
    import uuid

    import mlflow

    source = str(report["data_source"])
    use_experiment("ledgersentry-compare")
    deltas: dict[str, Mapping[str, Any]] = {}
    for key, d in report.get("paired_deltas_vs_incumbent", {}).items():
        deltas[str(d.get("variant", key))] = d
    tags = data_tags(source, data_dir)
    batch = uuid.uuid4().hex[:12]
    ids = []
    for row in report["results"]:
        variant = _variant(row)
        with mlflow.start_run(run_name=f"{experiment_suffix}:{variant}") as run:
            mlflow.set_tags({
                **tags, "comparison": experiment_suffix, "variant": variant,
                "comparison_batch": batch,
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
    readback = read_back_comparison(experiment_suffix, batch, report)
    if readback_path is not None:
        import json

        readback_path.write_text(json.dumps(readback, indent=2) + "\n")
    return ids


def read_back_comparison(experiment_suffix: str, batch: str,
                         report: Mapping[str, Any]) -> dict[str, Any]:
    """Read one comparison batch back from the store with mlflow.search_runs and
    require every logged metric and the selection tags to equal the report the
    script computed. Returns the deterministic part (no run ids, no fit times)."""
    import mlflow

    runs = mlflow.search_runs(experiment_names=["ledgersentry-compare"],
                              filter_string=f"tags.comparison_batch = '{batch}'",
                              output_format="list")
    by_variant = {r.data.tags["variant"]: r for r in runs}
    expected = {_variant(row): row for row in report["results"]}
    deltas = {str(d.get("variant", k)): d
              for k, d in report.get("paired_deltas_vs_incumbent", {}).items()}
    problems: list[str] = []
    if sorted(by_variant) != sorted(expected) or len(runs) != len(expected):
        problems.append(f"runs in store {sorted(by_variant)} != report {sorted(expected)}")
    out: dict[str, Any] = {}
    for variant in sorted(expected):
        r = by_variant.get(variant)
        if r is None:
            continue
        row = expected[variant]
        want = _scalars({k: v for k, v in row.items() if k not in ("max_iter", "learning_rate")})
        if variant in deltas:
            want.update(_scalars({k: v for k, v in deltas[variant].items()
                                  if k != "interval_excludes_zero"},
                                 "paired_delta_vs_incumbent."))
        got = dict(r.data.metrics)
        for k, v in want.items():
            if got.get(k) != v:
                problems.append(f"{variant} {k}: store {got.get(k)!r}, report {v!r}")
        extra = sorted(set(got) - set(want))
        if extra:
            problems.append(f"{variant}: store has metrics the report does not: {extra}")
        for tag, want_tag in (("is_incumbent", str(variant == report["incumbent"])),
                              ("selected_by_validation",
                               str(variant == report["selected_by_validation"]))):
            if r.data.tags.get(tag) != want_tag:
                problems.append(f"{variant} tag {tag}: {r.data.tags.get(tag)!r}")
        out[variant] = {
            "params": dict(sorted(r.data.params.items())),
            "metrics": {k: v for k, v in sorted(got.items())
                        if k not in NOT_DETERMINISTIC},
            "is_incumbent": r.data.tags["is_incumbent"] == "True",
            "selected_by_validation": r.data.tags["selected_by_validation"] == "True",
            "has_paired_delta": "paired_delta_excludes_zero" in r.data.tags,
        }
    if problems:
        raise ReadBackError("; ".join(problems))
    return {
        "what": (f"the {experiment_suffix} comparison as MLflow runs, read back from the "
                 "tracking store with mlflow.search_runs and checked value by value "
                 "against the report the script computed; fit_seconds is logged but "
                 "left out here because it is wall-clock time"),
        "experiment": "ledgersentry-compare",
        "data_source": str(report["data_source"]),
        "is_synthetic": bool(report["is_synthetic"]),
        "n_runs": len(runs),
        "n_runs_with_paired_delta": sum(v["has_paired_delta"] for v in out.values()),
        "incumbent": str(report["incumbent"]),
        "selected_by_validation": str(report["selected_by_validation"]),
        "runs": out,
    }
