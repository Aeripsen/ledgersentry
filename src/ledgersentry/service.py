"""
FastAPI serving layer for LedgerSentry.

GET  /health         -> liveness (always 200 while up; says if the model is loaded)
GET  /ready          -> readiness (200 only when the scorer can actually serve)
POST /predict        -> score one transaction, review_threshold as a request field
POST /predict/batch  -> score up to MAX_BATCH transactions in one vectorized call
POST /drift          -> PSI drift check of a window against the training reference
GET  /curve          -> the measured coverage-vs-precision curve (last train run)

Logging is structured (one JSON object per line) and metadata-only: decisions,
latencies, counts - never transaction feature values. Threat model, including
the joblib/pickle trust rule for the artifact: docs/threat_model.md.

Unlike a fixed-schema project, LedgerSentry's feature set is dataset-dependent
(the synthetic fixture has f_entity_daily_tx_count; Sparkov/IEEE-CIS/ULB would
each have different f_* columns, and some have no category at all). So /predict
does not hardcode the synthetic fixture's columns - it reads the exact expected
input columns straight off the FITTED preprocessor inside the loaded artifact,
which means this endpoint works with whichever dataset actually trained it.

Scoring goes through the compiled scorer (scoring.py): the fitted preprocessor
is compiled to a direct numpy transform once at load, because profiling showed
per-request pandas/ColumnTransformer work dominating single-row latency. The
compiled path is pinned byte-exact to the reference path by tests/test_scoring.py.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .config import get_settings
from .drain import DrainMiddleware
from .drift import drift_report
from .scoring import CompiledScorer, build_scorer

ARTIFACT_DIR = get_settings().artifact_dir
ARTIFACT = ARTIFACT_DIR / "ledgersentry.joblib"
MAX_BATCH = get_settings().max_batch  # request-size cap: bounds memory and latency


class _JsonFormatter(logging.Formatter):
    """One JSON object per line: machine-parseable, grep-able, no format drift."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level": record.levelname,
            "event": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))
        return json.dumps(entry)


logger = logging.getLogger("ledgersentry.service")
if not logger.handlers:  # don't stack handlers on reload/re-import
    _handler = logging.StreamHandler()
    _handler.setFormatter(_JsonFormatter())
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _log(event: str, **fields: object) -> None:
    """Structured log line. Policy: METADATA ONLY - decisions, latencies,
    counts. Never feature values: transaction fields are card-adjacent data
    and do not belong in server logs (see docs/threat_model.md)."""
    logger.info(event, extra={"fields": fields})

app = FastAPI(
    title="LedgerSentry",
    version=__version__,
    description="Real-time financial-transaction fraud detection with a tunable reject option.",
)
# Connection draining for rolling restarts (drain.py, deploy/k8s preStop).
app.add_middleware(DrainMiddleware)

_bundle = None
_scorer_cache: tuple[object, CompiledScorer] | None = None


def _load() -> dict[str, Any]:
    global _bundle
    if _bundle is None:
        if not ARTIFACT.exists():
            raise FileNotFoundError(
                "model artifact missing; run `python scripts/train.py` first"
            )
        _bundle = joblib.load(ARTIFACT)
    return _bundle


def _scorer() -> CompiledScorer:
    """Compiled scorer for the current bundle, built once per loaded bundle.
    Keyed by the bundle object itself (identity), so tests that swap the
    bundle get a fresh scorer and a stale one can never be served.

    The throwaway score after building pays sklearn's one-time first-predict
    cost (thread-pool spin-up, dispatch caches) at LOAD time: measured ~1.4 s
    on the first request without it, ~1 ms after. A payment-path service must
    never bill that to the first customer transaction."""
    global _scorer_cache
    bundle = _load()
    if _scorer_cache is None or _scorer_cache[0] is not bundle:
        scorer = build_scorer(bundle)
        scorer.score_one({})  # warmup: all-NaN row, result discarded
        _scorer_cache = (bundle, scorer)
    return _scorer_cache[1]


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


def _missing_fields(feats: dict, scorer: CompiledScorer) -> list[str]:
    """Expected model inputs absent from this request. Missing NUMERIC features
    reach the model as NaN, never a fake constant 0.0: the gradient-boosted
    model handles missing values natively (the repo's stated reason for choosing
    it), and 0.0 is a real, misleading value here - 0 is a legitimate amount or
    PCA component, not "this field was absent". Missing category defaults to
    'unknown' (the one-hot encoder was fit with handle_unknown='ignore'). The
    response surfaces what was imputed instead of hiding it."""
    return [c for c in scorer.numeric_cols + scorer.categorical_cols if c not in feats]


class Transaction(BaseModel):
    features: dict = Field(
        ...,
        description=(
            "Transaction fields: amount, category, entity_id, timestamp (or "
            "precomputed hour_of_day/day_of_week), plus any f_* numeric extras "
            "the loaded model expects. Missing numeric fields default to NaN "
            "(handled natively by the gradient-boosted model), missing category "
            "defaults to 'unknown'; the response lists any imputed fields under "
            "'missing_fields'."
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
def health() -> dict[str, str]:
    """Liveness + a status field: always 200 while the process is up, so an
    orchestrator does not kill a pod that merely lacks its artifact."""
    try:
        _load()
        return {"status": "ok", "model": "loaded", "version": __version__}
    except FileNotFoundError:
        return {"status": "degraded", "model": "missing", "version": __version__}


@app.get("/ready")
def ready() -> dict[str, Any]:
    """Readiness: 200 only when the artifact is loaded and the compiled scorer
    is built - the binary gate for routing traffic (versus /health, which is
    liveness and stays 200 while degraded)."""
    try:
        scorer = _scorer()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return {
        "status": "ready",
        "version": __version__,
        "expected_features": len(scorer.numeric_cols) + len(scorer.categorical_cols),
    }


class BatchRequest(BaseModel):
    transactions: list[dict] = Field(
        ...,
        max_length=MAX_BATCH,
        description=f"Feature dicts, same shape as /predict's 'features'. Max {MAX_BATCH}.",
    )
    review_threshold: float = Field(0.0, ge=0.0, le=1.0)


@app.post("/predict")
def predict(req: Transaction) -> dict[str, Any]:
    try:
        scorer = _scorer()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    feats = _derive_time_features(req.features)
    t0 = time.perf_counter()
    try:
        result = scorer.score_one(feats, review_threshold=req.review_threshold)
    except (TypeError, ValueError) as e:
        raise HTTPException(
            status_code=422, detail=f"non-numeric value for a numeric feature: {e}"
        ) from e
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    missing = _missing_fields(feats, scorer)
    _log(
        "predict",
        decision=result.decision,
        p_fraud=round(result.p_fraud, 4),
        latency_ms=latency_ms,
        n_missing=len(missing),
    )
    return {
        "decision": result.decision,
        "p_fraud": round(result.p_fraud, 4),
        "confidence": round(result.confidence, 4),
        "missing_fields": missing,
        "latency_ms": latency_ms,
    }


@app.post("/predict/batch")
def predict_batch(req: BatchRequest) -> dict[str, Any]:
    """Vectorized scoring: one transform + one model call for the whole batch.
    This is the throughput path (see artifacts/benchmark.json); /predict is the
    latency path. Returns per-row decisions plus which expected model inputs
    were absent from the batch as a whole."""
    try:
        scorer = _scorer()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    if not req.transactions:
        return {"n": 0, "results": [], "missing_columns": []}
    rows = [_derive_time_features(t) for t in req.transactions]
    frame = pd.DataFrame(rows)
    t0 = time.perf_counter()
    try:
        batch = scorer.score_frame(frame, review_threshold=req.review_threshold)
    except (TypeError, ValueError) as e:
        raise HTTPException(
            status_code=422, detail=f"non-numeric value for a numeric feature: {e}"
        ) from e
    _log(
        "predict_batch",
        n=len(rows),
        n_fraud=int((batch.decisions == "fraud").sum()),
        n_review=int((batch.decisions == "review").sum()),
        latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
    )
    expected = scorer.numeric_cols + scorer.categorical_cols
    return {
        "n": len(rows),
        "results": [
            {
                "decision": str(batch.decisions[i]),
                "p_fraud": round(float(batch.p_fraud[i]), 4),
                "confidence": round(float(batch.confidence[i]), 4),
            }
            for i in range(len(rows))
        ],
        "missing_columns": [c for c in expected if c not in frame.columns],
    }


@app.post("/drift")
def drift(req: BatchRequest) -> dict[str, Any]:
    """PSI drift check of a window of recent transactions against the training
    reference frozen inside the artifact (see drift.py: it flags marginal
    feature shift, not model wrongness - treat alerts as a trigger to look).
    Send the same feature dicts /predict/batch takes; review_threshold is
    ignored here."""
    try:
        bundle = _load()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    reference = bundle.get("drift_reference")
    if not reference:
        raise HTTPException(
            status_code=503,
            detail="artifact has no drift reference; retrain with scripts/train.py",
        )
    if not req.transactions:
        raise HTTPException(status_code=422, detail="empty window")
    cfg = get_settings()
    window = pd.DataFrame([_derive_time_features(t) for t in req.transactions])
    try:
        report = drift_report(
            reference, window, psi_watch=cfg.psi_watch, psi_alert=cfg.psi_alert
        )
    except (TypeError, ValueError) as e:
        raise HTTPException(
            status_code=422, detail=f"non-numeric value for a numeric feature: {e}"
        ) from e
    _log(
        "drift",
        n_window=report["n_window"],
        worst_feature=report["worst_feature"],
        worst_psi=report["worst_psi"],
        n_alerts=report["n_alerts"],
    )
    return report


@app.get("/curve")
def curve() -> dict[str, Any]:
    path = ARTIFACT_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail="no metrics; run training first")
    return json.loads(path.read_text())
