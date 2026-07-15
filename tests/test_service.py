import pandas as pd
from fastapi.testclient import TestClient

from ledgersentry import service
from ledgersentry.data import build_preprocessor, engineer_time_features
from ledgersentry.model import FraudDetector


def _tiny_bundle():
    """Train a small real model on a handful of synthetic-shaped rows (no
    network, no reliance on the big 8,000-row fixture)."""
    categories = ["groceries", "electronics", "travel"]
    rows = []
    for i in range(60):
        rows.append(
            {
                "transaction_id": f"tx_{i}",
                "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=i),
                "entity_id": f"acct_{i % 5}",
                "amount": float(10 + i * 3),
                "category": categories[i % 3],
                "is_fraud": int(i % 10 == 0),
                "f_entity_daily_tx_count": i % 4,
            }
        )
    df = pd.DataFrame(rows)
    df = engineer_time_features(df)
    pre = build_preprocessor(df)
    X = pre.fit_transform(df)
    model = FraudDetector(max_iter=20).fit(X, df["is_fraud"].to_numpy())
    return {"preprocessor": pre, "model": model}


def test_health_ok():
    client = TestClient(service.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] in {"ok", "degraded"}


def test_predict_returns_decision(monkeypatch):
    monkeypatch.setattr(service, "_bundle", _tiny_bundle())
    client = TestClient(service.app)
    resp = client.post(
        "/predict",
        json={
            "features": {
                "amount": 42.0,
                "category": "electronics",
                "hour_of_day": 2,
                "day_of_week": 1,
            }
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] in {"fraud", "legit", "review"}
    assert 0.0 <= body["p_fraud"] <= 1.0
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_missing_fields_default_cleanly(monkeypatch):
    """Only amount given - missing numeric f_* default to 0, missing category
    defaults to 'unknown', and the endpoint must not 500 on a sparse request."""
    monkeypatch.setattr(service, "_bundle", _tiny_bundle())
    client = TestClient(service.app)
    resp = client.post("/predict", json={"features": {"amount": 100.0}})
    assert resp.status_code == 200
    assert resp.json()["decision"] in {"fraud", "legit", "review"}


def test_predict_review_threshold_plumbing(monkeypatch):
    monkeypatch.setattr(service, "_bundle", _tiny_bundle())
    client = TestClient(service.app)
    payload = {
        "features": {
            "amount": 42.0,
            "category": "electronics",
            "hour_of_day": 2,
            "day_of_week": 1,
        }
    }

    base = client.post("/predict", json={**payload, "review_threshold": 0.0}).json()
    assert base["decision"] != "review"

    hi = client.post("/predict", json={**payload, "review_threshold": 1.0}).json()
    if base["confidence"] < 1.0:
        # confidence below the bar -> abstain
        assert hi["decision"] == "review"
    else:
        # a fully certain model cannot be rejected past its own certainty
        assert hi["decision"] == base["decision"]


def test_curve_without_metrics_is_503(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "ARTIFACT_DIR", tmp_path)
    client = TestClient(service.app)
    resp = client.get("/curve")
    assert resp.status_code == 503
