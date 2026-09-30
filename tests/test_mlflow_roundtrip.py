"""MLflow for real, not a string test: train on the synthetic fixture, log to a
throwaway SQLite store, read the runs back, reload the model, and show that a
disagreement stops the run instead of only tagging it.

conftest.py switches tracking off suite-wide; these tests switch it back on
against their own temporary store, so nothing reaches the developer's mlflow.db.
Skipped when mlflow is not installed (the base CI job runs without it on purpose,
to prove the seam is optional); the CI mlops job installs mlflow and runs this."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

mlflow = pytest.importorskip("mlflow")

from ledgersentry import tracking  # noqa: E402
from ledgersentry.config import get_settings  # noqa: E402
from ledgersentry.data import (  # noqa: E402
    build_preprocessor,
    engineer_time_features,
    make_synthetic,
    temporal_grouped_split,
)
from ledgersentry.model import FraudDetector  # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'store.db').as_posix()}")
    monkeypatch.setenv(tracking.SWITCH, "1")
    monkeypatch.setenv("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    return tmp_path


def _fitted(model_name: str = "hist_gbdt"):
    df = engineer_time_features(make_synthetic(n_rows=3000, n_entities=150, days=30))
    train_df, test_df = temporal_grouped_split(df, test_size=0.2)
    pre = build_preprocessor(train_df)
    model = FraudDetector(max_iter=20, model=model_name).fit(
        pre.fit_transform(train_df), train_df["is_fraud"].to_numpy())
    return pre, model, test_df


def _log(tmp_path, model_name: str = "hist_gbdt", offset: float = 0.0):
    pre, model, test_df = _fitted(model_name)
    p = model.predict_proba_fraud(pre.transform(test_df))
    metrics = {"data_source": "synthetic", "model": model_name, "pr_auc": 0.91,
               "recall_at_full_coverage": 0.8,
               "demoted_metrics": {"roc_auc": 0.99, "why": "text is not a metric"},
               "coverage_precision_curve": [
                   {"review_threshold": 0.5, "coverage": 1.0, "recall_auto": 0.8},
                   {"review_threshold": 0.9, "coverage": 0.7, "recall_auto": 0.6}]}
    mpath = tmp_path / "metrics_synthetic.json"
    mpath.write_text(json.dumps(metrics))
    cfg = get_settings()
    settings = SimpleNamespace(data_dir=tmp_path, artifact_dir=tmp_path,
                               random_state=cfg.random_state, max_iter=20,
                               learning_rate=cfg.learning_rate, test_size=0.2,
                               review_thresholds=cfg.review_thresholds)
    return tracking.log_training_run(settings=settings, metrics=metrics, preprocessor=pre,
                                     model=model, X_test=test_df, p_fraud_test=p + offset,
                                     metrics_path=mpath), (pre, model, test_df)


def test_training_run_is_stored_and_its_model_reloads_identically(store):
    (run_id, same), (pre, model, test_df) = _log(store)
    assert same is True
    run = mlflow.get_run(run_id)
    assert run.data.tags["is_synthetic"] == "True"
    assert run.data.tags["model_serialization"] == "skops"
    assert run.data.tags["roundtrip_predictions_identical"] == "True"
    assert run.data.params["model"] == "hist_gbdt"
    assert run.data.metrics["pr_auc"] == 0.91
    assert run.data.metrics["demoted.roc_auc"] == 0.99
    assert run.data.metrics["roundtrip.n_rows"] == len(test_df)
    hist = mlflow.MlflowClient().get_metric_history(run_id, "curve.recall_auto")
    assert sorted((m.step, m.value) for m in hist) == [(50, 0.8), (90, 0.6)]
    # an independent reload from the store, through skops and the trusted list
    (logged,) = mlflow.search_logged_models(filter_string=f"source_run_id = '{run_id}'",
                                            output_format="list")
    loaded = mlflow.sklearn.load_model(logged.model_uri)
    assert np.array_equal(loaded[-1].predict_proba_fraud(loaded[:-1].transform(test_df)),
                          model.predict_proba_fraud(pre.transform(test_df)))


def test_a_model_that_does_not_reproduce_stops_the_run(store):
    with pytest.raises(tracking.RoundTripError):
        _log(store, offset=1e-15)
    (run,) = mlflow.search_runs(experiment_names=["ledgersentry-train"], output_format="list")
    assert run.data.tags["roundtrip_predictions_identical"] == "False"


def test_serialization_is_chosen_by_type_not_by_fallback():
    _, model, _ = _fitted("hist_gbdt")
    assert tracking.serialization_for(model) == "skops"
    _, model, _ = _fitted("logreg")
    assert tracking.serialization_for(model) == "skops"


def test_lgbm_is_the_only_cloudpickle_model(store):
    pytest.importorskip("lightgbm")
    (run_id, same), _ = _log(store, model_name="lgbm")
    assert same is True
    assert mlflow.get_run(run_id).data.tags["model_serialization"].startswith("cloudpickle")


def _report():
    rows = [{"feature_set": "base", "config": c, "val_pr_auc": v, "test_pr_auc": t,
             "fit_seconds": 1.5, "max_iter": 200, "learning_rate": 0.1}
            for c, v, t in (("gbdt_default", 0.70, 0.72), ("gbdt_shallow", 0.75, 0.71))]
    return {"data_source": "synthetic", "is_synthetic": True,
            "incumbent": "base|gbdt_default", "selected_by_validation": "base|gbdt_shallow",
            "selection_rule": "highest val_pr_auc", "split": {"n_train": 10, "note": "x"},
            "results": rows,
            "paired_deltas_vs_incumbent": {"base|gbdt_shallow": {
                "variant": "base|gbdt_shallow", "delta_pr_auc": -0.01, "ci_lower": -0.05,
                "ci_upper": 0.03, "interval_excludes_zero": False}}}


def test_comparison_runs_are_read_back_and_written(store):
    out = store / "readback.json"
    ids = tracking.log_comparison("compare", _report(), data_dir=store, readback_path=out)
    rb = json.loads(out.read_text())
    assert len(ids) == rb["n_runs"] == 2
    assert rb["n_runs_with_paired_delta"] == 1
    shallow = rb["runs"]["base|gbdt_shallow"]
    assert shallow["selected_by_validation"] is True and shallow["has_paired_delta"] is True
    assert shallow["metrics"]["paired_delta_vs_incumbent.delta_pr_auc"] == -0.01
    assert "fit_seconds" not in shallow["metrics"]  # wall clock, kept out of the diffed file
    assert rb["runs"]["base|gbdt_default"]["is_incumbent"] is True


def test_a_store_that_disagrees_with_the_report_is_caught(store, monkeypatch):
    real = mlflow.log_metrics

    def corrupt(metrics, *a, **k):  # the store receives a different test PR-AUC
        return real({name: (v + 0.5 if name == "test_pr_auc" else v)
                     for name, v in metrics.items()}, *a, **k)

    monkeypatch.setattr(mlflow, "log_metrics", corrupt)
    with pytest.raises(tracking.ReadBackError):
        tracking.log_comparison("compare", _report(), data_dir=store)
