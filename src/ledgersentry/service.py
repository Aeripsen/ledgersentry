"""
FastAPI serving layer for LedgerSentry.

GET  /health   -> liveness + whether a trained model is loaded
POST /predict  -> score one transaction, review_threshold as a request field
GET  /curve    -> the measured coverage-vs-precision curve (from the last train run)

Unlike a fixed-schema project, LedgerSentry's feature set is dataset-dependent
(the synthetic fixture has f_entity_daily_tx_count; Sparkov/IEEE-CIS/ULB would
each have different f_* columns, and some have no category at all). So /predict
does not hardcode the synthetic fixture's columns - it reads the exact expected
input columns straight off the FITTED preprocessor inside the loaded artifact,
which means this endpoint works with whichever dataset actually trained it.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
ARTIFACT = ARTIFACT_DIR / "ledgersentry.joblib"

app = FastAPI(
    title="LedgerSentry",
    version=__version__,
    description="Real-time financial-transaction fraud detection with a tunable reject option.",
)

_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        if not ARTIFACT.exists():
            raise FileNotFoundError(
                "model artifact missing; run `python scripts/train.py` first"
            )
        _bundle = joblib.load(ARTIFACT)
    return _bundle


def _expected_columns(preprocessor) -> tuple[list[str], list[str]]:
    """(numeric_cols, categorical_cols) the FITTED preprocessor actually expects,
    pulled from `transformers_` rather than hardcoded - see module docstring."""
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for name, _trans, cols in preprocessor.transformers_:
        if name == "num":
            numeric_cols = list(cols)
        elif name == "cat":
            categorical_cols = list(cols)
    return numeric_cols, categorical_cols


def _derive_time_features(features: dict) -> dict:
    """If a raw `timestamp` was given and hour_of_day/day_of_week weren't,
    derive them (data.engineer_time_features does the same thing at train
    time). Explicit hour_of_day/day_of_week in the request always win."""
    feats = dict(features)
    ts = feats.get("timestamp")
    if ts is not None and ("hour_of_day" not in feats or "day_of_week" not in feats):
        parsed = pd.Timestamp(ts)
        feats.setdefault("hour_of_day", parsed.hour)
        feats.setdefault("day_of_week", parsed.dayofweek)
    return feats


def _row(features: dict, numeric_cols: list[str], categorical_cols: list[str]) -> pd.DataFrame:
    feats = _derive_time_features(features)
    row = {c: feats.get(c, 0) for c in numeric_cols}
    for c in categorical_cols:
        row[c] = feats.get(c, "unknown")
    return pd.DataFrame([row])


class Transaction(BaseModel):
    features: dict = Field(
        ...,
        description=(
            "Transaction fields: amount, category, entity_id, timestamp (or "
            "precomputed hour_of_day/day_of_week), plus any f_* numeric extras "
            "the loaded model expects. Missing numeric fields default to 0, "
            "missing category defaults to 'unknown'."
        ),
        examples=[
            {
                "amount": 812.50,
                "category": "electronics",
                "entity_id": "acct_0007",
                "timestamp": "2026-01-15T02:14:00",
                "f_entity_daily_tx_count": 3,
            }
        ],
    )
    review_threshold: float = Field(
        0.0, ge=0.0, le=1.0,
        description=(
            "Below this confidence the decision becomes 'review' instead of a "
            "forced fraud/legit call."
        ),
    )


@app.get("/health")
def health():
    try:
        _load()
        return {"status": "ok", "model": "loaded", "version": __version__}
    except FileNotFoundError:
        return {"status": "degraded", "model": "missing", "version": __version__}


@app.post("/predict")
def predict(req: Transaction):
    try:
        bundle = _load()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    numeric_cols, categorical_cols = _expected_columns(bundle["preprocessor"])
    X = bundle["preprocessor"].transform(_row(req.features, numeric_cols, categorical_cols))
    decision, p_fraud, confidence = bundle["model"].decide(X, review_threshold=req.review_threshold)
    return {
        "decision": str(decision[0]),
        "p_fraud": round(float(p_fraud[0]), 4),
        "confidence": round(float(confidence[0]), 4),
    }


@app.get("/curve")
def curve():
    path = ARTIFACT_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail="no metrics; run training first")
    return json.loads(path.read_text())
