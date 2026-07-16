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
    """Only amount given - missing numeric f_* default to NaN (not a fake 0),
    missing category defaults to 'unknown', the endpoint must not 500 on a sparse
    request, and the response lists what it imputed under 'missing_fields'."""
    monkeypatch.setattr(service, "_bundle", _tiny_bundle())
    client = TestClient(service.app)
    resp = client.post("/predict", json={"features": {"amount": 100.0}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] in {"fraud", "legit", "review"}
    # hour_of_day/day_of_week (no timestamp), f_entity_daily_tx_count, and category
    # were all absent and must be reported back, not silently zero-filled.
    assert set(body["missing_fields"]) >= {
        "hour_of_day", "day_of_week", "f_entity_daily_tx_count", "category",
    }


def test_missing_numeric_defaults_to_nan_not_zero(monkeypatch):
    """Guard the honesty rule: an absent numeric field must reach the model as NaN
    (which HistGradientBoosting handles), never a misleading real 0.0."""
    import numpy as np

    monkeypatch.setattr(service, "_bundle", _tiny_bundle())
    scorer = service._scorer()
    feats = {"amount": 100.0}
    assert "f_entity_daily_tx_count" in service._missing_fields(feats, scorer)
    vec = scorer.transform_one(feats)
    k = scorer.numeric_cols.index("f_entity_daily_tx_count")
    assert np.isnan(vec[0, scorer._num_offset + k])
    # a provided field is untouched
    a = scorer.numeric_cols.index("amount")
    assert vec[0, scorer._num_offset + a] == 100.0


def test_predict_batch_matches_single(monkeypatch):
    """The vectorized batch path must agree with /predict row by row."""
    monkeypatch.setattr(service, "_bundle", _tiny_bundle())
    client = TestClient(service.app)
    transactions = [
        {"amount": 42.0, "category": "electronics", "hour_of_day": 2, "day_of_week": 1},
        {"amount": 900.0, "category": "travel", "hour_of_day": 3, "day_of_week": 5},
        {"amount": 12.0},  # sparse row: missing fields become NaN/'unknown'
    ]
    batch = client.post(
        "/predict/batch",
        json={"transactions": transactions, "review_threshold": 0.0},
    )
    assert batch.status_code == 200
    body = batch.json()
    assert body["n"] == 3
    for tx, row in zip(transactions, body["results"], strict=True):
        single = client.post(
            "/predict", json={"features": tx, "review_threshold": 0.0}
        ).json()
        assert row["decision"] == single["decision"]
        assert row["p_fraud"] == single["p_fraud"]


def test_predict_batch_empty_and_cap(monkeypatch):
    monkeypatch.setattr(service, "_bundle", _tiny_bundle())
    client = TestClient(service.app)
    empty = client.post("/predict/batch", json={"transactions": []})
    assert empty.status_code == 200
    assert empty.json()["n"] == 0
    over_cap = client.post(
        "/predict/batch",
        json={"transactions": [{"amount": 1.0}] * (service.MAX_BATCH + 1)},
    )
    assert over_cap.status_code == 422  # pydantic max_length, not a 500


def test_predict_non_numeric_value_is_422_not_500(monkeypatch):
    monkeypatch.setattr(service, "_bundle", _tiny_bundle())
    client = TestClient(service.app)
    resp = client.post(
        "/predict", json={"features": {"amount": "not-a-number"}}
    )
    assert resp.status_code == 422


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
