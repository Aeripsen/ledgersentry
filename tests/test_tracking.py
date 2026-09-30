"""The MLflow seam and the committed Evidently summary. Runs without mlflow or
evidently installed: tracking must stay optional, and the committed summary must
agree with the metrics and comparison files every published number comes from."""
from __future__ import annotations

import json
from pathlib import Path

from ledgersentry import tracking

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def test_metric_key_drops_illegal_chars():
    assert tracking.metric_key("paired_delta_vs_incumbent.ci_lower") == (
        "paired_delta_vs_incumbent.ci_lower"
    )
    assert tracking.metric_key("a:b(c)") == "a_b_c_"


def test_switch_turns_tracking_off(monkeypatch):
    monkeypatch.setenv(tracking.SWITCH, "0")
    assert tracking.enabled() is False


def test_tracking_uri_defaults_to_repo_sqlite_and_respects_env(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert tracking.tracking_uri().startswith("sqlite:///")
    assert tracking.tracking_uri().endswith("/mlflow.db")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    assert tracking.tracking_uri() == "http://127.0.0.1:5000"


def test_synthetic_runs_are_tagged_synthetic(tmp_path):
    tags = tracking.data_tags("synthetic", tmp_path)
    assert tags["is_synthetic"] == "True" and tags["data_source"] == "synthetic"


def test_evidently_windows_are_the_published_numbers():
    ev = json.loads((ARTIFACTS / "evidently_summary_ulb_creditcard.json").read_text())
    m = json.loads((ARTIFACTS / "metrics_ulb_creditcard.json").read_text())
    comp = json.loads((ARTIFACTS / "comparison_ulb_creditcard.json").read_text())
    incumbent = next(
        r for r in comp["results"] if f"{r['feature_set']}|{r['config']}" == comp["incumbent"]
    )
    pr = ev["performance"]["pr_auc"]
    assert pr["current"] == m["pr_auc"]
    assert pr["reference"] == incumbent["val_pr_auc"]
    assert ev["current_window"]["rows"] == m["n_test"]
    assert ev["current_window"]["fraud"] == m["n_test_fraud"]
    assert all(ev["checks_against_committed_artifacts"].values())
    assert (Path(__file__).resolve().parents[1] / ev["html_report"]).exists()
