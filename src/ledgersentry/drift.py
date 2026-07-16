"""
Feature-drift detection: PSI of a scored window against the training data.

Fraud drifts fast and ADVERSARIALLY - attackers actively probe for the model's
blind spots, so the feature distribution moving is the expected case, not the
exception. PSI (Population Stability Index, the standard credit-risk stability
metric) on each feature's marginal distribution is the cheap, always-on first
line: it catches the shifts that silently rot a model between retrains.

Be clear about what it is NOT: PSI watches marginals, so it misses joint-
distribution shifts and says nothing about label drift (fraudsters changing
behavior WITHIN the same feature ranges). It flags "the world your model sees
has changed"; only fresh labels can say "and the model is now wrong". Treat an
alert as a trigger to look, not as proof of damage.

Mechanics: at train time, train.py stores per-feature quantile bin edges and
bin proportions from the TRAIN split inside the model artifact (the reference
is frozen with the model it describes). At serve/monitor time, drift_report
bins a window of recent transactions with those same edges and computes

    PSI = sum_bins (p_window - p_ref) * ln(p_window / p_ref)

plus the null-fraction shift per feature (a missing-data spike is drift too,
and PSI over non-null values alone would hide it). Thresholds follow the
widely used credit-scoring convention - <0.1 stable, 0.1-0.25 watch, >=0.25
alert - which is a rule of thumb, not a law; both knobs live in config.

Surfaces: POST /drift on the service (send a window of transactions, get
per-feature PSI + flags) and drift_report() for offline use.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DEFAULT_BINS = 10
_EPS = 1e-4  # floor for bin proportions so empty bins don't blow up the log


def psi(
    reference_proportions: np.ndarray, window_proportions: np.ndarray
) -> float:
    """PSI between two binned proportion vectors (same bin edges)."""
    p_ref = np.clip(np.asarray(reference_proportions, dtype=float), _EPS, None)
    p_win = np.clip(np.asarray(window_proportions, dtype=float), _EPS, None)
    p_ref = p_ref / p_ref.sum()
    p_win = p_win / p_win.sum()
    return float(np.sum((p_win - p_ref) * np.log(p_win / p_ref)))


def _bin_proportions(values: np.ndarray, inner_edges: np.ndarray) -> np.ndarray:
    """Bin with (-inf, *inner_edges, +inf) so values outside the training range
    land in the outer bins instead of vanishing - out-of-range mass IS drift."""
    edges = np.concatenate(([-np.inf], inner_edges, [np.inf]))
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    return counts / total if total else counts.astype(float)


def reference_stats(
    df: pd.DataFrame, columns: list[str], bins: int = DEFAULT_BINS
) -> dict[str, Any]:
    """Frozen training-time reference for each numeric feature: inner quantile
    bin edges, the training bin proportions, and the training null fraction.
    Stored inside the model artifact so reference and model can never drift
    apart from each other."""
    out: dict[str, Any] = {}
    for col in columns:
        values = df[col].to_numpy(dtype=float)
        null_fraction = float(np.isnan(values).mean())
        finite = values[~np.isnan(values)]
        if finite.size == 0:
            continue  # a feature that is all-NaN in training has no distribution to track
        # unique() collapses duplicate quantiles (discrete features like
        # hour_of_day yield fewer, wider bins - still valid)
        inner_edges = np.unique(np.quantile(finite, np.linspace(0, 1, bins + 1)[1:-1]))
        out[col] = {
            "inner_edges": [float(e) for e in inner_edges],
            "proportions": [float(p) for p in _bin_proportions(finite, inner_edges)],
            "null_fraction": null_fraction,
            "n_reference": int(finite.size),
        }
    return out


def drift_report(
    reference: dict[str, Any],
    window: pd.DataFrame,
    psi_watch: float,
    psi_alert: float,
) -> dict[str, Any]:
    """Per-feature PSI of `window` against the stored training reference.
    Thresholds are explicit arguments (config owns the defaults): a monitoring
    surface with baked-in sensitivity is a monitoring surface someone forgot
    they never tuned."""
    features: dict[str, Any] = {}
    worst: tuple[str, float] | None = None
    for col, ref in reference.items():
        if col not in window.columns:
            features[col] = {"status": "missing_from_window"}
            continue
        values = window[col].to_numpy(dtype=float)
        null_fraction = float(np.isnan(values).mean()) if values.size else 0.0
        finite = values[~np.isnan(values)]
        if finite.size == 0:
            features[col] = {"status": "all_null_in_window", "null_fraction": 1.0}
            continue
        proportions = _bin_proportions(finite, np.asarray(ref["inner_edges"]))
        value = psi(np.asarray(ref["proportions"]), proportions)
        status = "stable"
        if value >= psi_alert:
            status = "alert"
        elif value >= psi_watch:
            status = "watch"
        features[col] = {
            "psi": round(value, 4),
            "status": status,
            "null_fraction": round(null_fraction, 4),
            "null_fraction_reference": round(ref["null_fraction"], 4),
        }
        if worst is None or value > worst[1]:
            worst = (col, value)

    return {
        "n_window": int(len(window)),
        "psi_watch": psi_watch,
        "psi_alert": psi_alert,
        "worst_feature": worst[0] if worst else None,
        "worst_psi": round(worst[1], 4) if worst else None,
        "n_alerts": sum(1 for f in features.values() if f.get("status") == "alert"),
        "features": features,
    }
