"""
Drift detection must earn its endpoint: a same-distribution window stays quiet,
a shifted window screams on the shifted feature only, and the reference is
frozen into the artifact beside the model it describes.
"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ledgersentry import data, drift, service
from ledgersentry.data import build_preprocessor, engineer_time_features
from ledgersentry.model import FraudDetector

WATCH, ALERT = 0.1, 0.25


@pytest.fixture(scope="module")
def train_df():
    return engineer_time_features(data.make_synthetic(n_rows=5000, seed=21))


@pytest.fixture(scope="module")
def reference(train_df):
    numeric, _ = data.feature_columns(train_df)
    return drift.reference_stats(train_df, numeric)


def test_reference_covers_numeric_features(reference, train_df):
    numeric, _ = data.feature_columns(train_df)
    assert set(reference) == set(numeric)
    for ref in reference.values():
        assert ref["n_reference"] > 0
        assert abs(sum(ref["proportions"]) - 1.0) < 1e-9


def test_same_distribution_is_stable(reference, train_df):
    # a different seeded draw from the SAME generator: honest "no drift" window
    window = engineer_time_features(data.make_synthetic(n_rows=3000, seed=99))
    report = drift.drift_report(reference, window, psi_watch=WATCH, psi_alert=ALERT)
    assert report["n_alerts"] == 0
    assert report["features"]["amount"]["status"] == "stable"


def test_shifted_feature_alerts_and_only_it(reference, train_df):
    window = engineer_time_features(data.make_synthetic(n_rows=3000, seed=99))
    window["amount"] = window["amount"] * 10.0  # a 10x amount shift, unmissable
    report = drift.drift_report(reference, window, psi_watch=WATCH, psi_alert=ALERT)
    assert report["features"]["amount"]["status"] == "alert"
    assert report["worst_feature"] == "amount"
    assert report["features"]["hour_of_day"]["status"] == "stable"


def test_out_of_range_mass_is_seen(reference):
    """Values beyond the training range must land in the open outer bins and
    register as drift, not silently vanish."""
    window = pd.DataFrame({"amount": np.full(500, 1e9)})
    report = drift.drift_report(reference, window, psi_watch=WATCH, psi_alert=ALERT)
    assert report["features"]["amount"]["status"] == "alert"


def test_null_spike_is_reported(reference):
    window = pd.DataFrame({"amount": [np.nan] * 400 + [50.0] * 100})
    report = drift.drift_report(reference, window, psi_watch=WATCH, psi_alert=ALERT)
    assert report["features"]["amount"]["null_fraction"] == 0.8
    assert report["features"]["amount"]["null_fraction_reference"] == 0.0


def test_psi_zero_for_identical_proportions():
    p = np.array([0.2, 0.3, 0.5])
    assert drift.psi(p, p) == pytest.approx(0.0, abs=1e-12)


def test_train_freezes_reference_into_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("LEDGERSENTRY_DATA_DIR", str(tmp_path / "no_data"))
    monkeypatch.setenv("LEDGERSENTRY_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("LEDGERSENTRY_MAX_ITER", "20")
    import joblib

    from ledgersentry import config, train

    config.reset_settings()
    try:
        train.main()
    finally:
        config.reset_settings()
    bundle = joblib.load(tmp_path / "artifacts" / "ledgersentry.joblib")
    assert "drift_reference" in bundle
    assert "amount" in bundle["drift_reference"]


def _bundle_with_reference(train_df, reference):
    pre = build_preprocessor(train_df)
    X = pre.fit_transform(train_df)
    model = FraudDetector(max_iter=20).fit(X, train_df["is_fraud"].to_numpy())
    return {"preprocessor": pre, "model": model, "drift_reference": reference}


def test_drift_endpoint_flags_shift(monkeypatch, train_df, reference):
    monkeypatch.setattr(service, "_bundle", _bundle_with_reference(train_df, reference))
    client = TestClient(service.app)
    stable = [{"amount": float(a)} for a in train_df["amount"].head(500)]
    resp = client.post("/drift", json={"transactions": stable})
    assert resp.status_code == 200
    assert resp.json()["features"]["amount"]["status"] == "stable"

    shifted = [{"amount": float(a) * 10.0} for a in train_df["amount"].head(500)]
    resp = client.post("/drift", json={"transactions": shifted})
    assert resp.json()["features"]["amount"]["status"] == "alert"


def test_drift_endpoint_without_reference_is_503(monkeypatch, train_df):
    bundle = _bundle_with_reference(train_df, {})
    del bundle["drift_reference"]
    monkeypatch.setattr(service, "_bundle", bundle)
    client = TestClient(service.app)
    resp = client.post("/drift", json={"transactions": [{"amount": 5.0}]})
    assert resp.status_code == 503


def test_drift_endpoint_empty_window_is_422(monkeypatch, train_df, reference):
    monkeypatch.setattr(service, "_bundle", _bundle_with_reference(train_df, reference))
    client = TestClient(service.app)
    resp = client.post("/drift", json={"transactions": []})
    assert resp.status_code == 422
