"""
Baseline fraud classifier with a tunable reject (abstain -> "review") knob.

Default model: `HistGradientBoostingClassifier` (gradient-boosted trees,
scikit-learn). Chosen over xgboost because it ships inside scikit-learn -
already a dependency, so it installs cleanly everywhere including CI with no
compiled-wheel risk - and it natively handles missing values in numeric
features, which matters once a real dataset with sparse anonymized columns
(ULB's V1..V28, IEEE-CIS's C/D/V columns) is dropped in. The classifier is
resolved through registry.py: `logreg` ships as the tested linear baseline,
and a new model is one register() call, never an edit here.

Imbalance handling: this baseline uses balanced SAMPLE WEIGHTS (computed from the
train split's own class counts), not resampling (SMOTE/undersampling). That is a
deliberate choice, not a missing feature: weighting never touches X, so it can't
suffer the classic SMOTE-before-split leakage bug (synthetic minority points
generated from the full set leaking into the test fold). If a
future run wants resampling, fit it on the train split only, exactly the same
rule that applies to the sample weights here.

The reject knob is the same idea as FlowSentry's two-stage reject option (itself
from Sepehr Jafari's SECRYPT 2026 paper), translated to binary fraud scoring:
instead of a top-class confidence, we use distance from the decision boundary,
`max(p_fraud, 1 - p_fraud)`, which ranges 0.5 (totally unsure) to 1.0 (certain).
Below a chosen threshold the system abstains and returns "review" instead of
forcing a fraud/legit guess - exactly what a fraud operations desk does with an
uncertain transaction.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from . import registry

FRAUD = "fraud"
LEGIT = "legit"
REVIEW = "review"


class FraudDetector(BaseEstimator):
    """`model` is a registry name (see registry.py; `hist_gbdt` is the default
    behind every committed number, `logreg` the linear baseline). The reject
    knob, the balanced weighting, and the curve logic below are model-agnostic:
    swapping the classifier never touches them."""

    def __init__(
        self,
        random_state: int = 42,
        max_iter: int = 200,
        learning_rate: float = 0.1,
        model: str = "hist_gbdt",
    ):
        self.random_state = random_state
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.model = model

    def fit(self, X: ArrayLike, y: ArrayLike) -> FraudDetector:
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)

        # Balanced sample weights, same formula as sklearn's class_weight="balanced"
        # (n_samples / (n_classes * count_per_class)), computed from TRAIN labels
        # only. Passed as sample_weight so the same mechanism works for every
        # registered model instead of depending on per-estimator class_weight support.
        counts = np.bincount(y, minlength=int(y.max()) + 1)
        weight_per_class = counts.sum() / (len(counts) * np.maximum(counts, 1))
        sample_weight = weight_per_class[y]

        self.model_ = registry.create(
            self.model, self.random_state, self.max_iter, self.learning_rate
        )
        if isinstance(self.model_, Pipeline):
            # route the weights to the Pipeline's final step (sklearn's own
            # step-prefixed fit-param convention)
            final_step = self.model_.steps[-1][0]
            self.model_.fit(X, y, **{f"{final_step}__sample_weight": sample_weight})
        else:
            self.model_.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba_fraud(self, X: ArrayLike) -> np.ndarray:
        """P(fraud) per row - the positive-class column of predict_proba."""
        proba = self.model_.predict_proba(X)
        fraud_col = list(self.model_.classes_).index(1)
        return np.asarray(proba[:, fraud_col])

    def decide(
        self, X: ArrayLike, review_threshold: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (decision, p_fraud, confidence).
        decision[i] in {"fraud", "legit", "review"}. confidence = max(p_fraud,
        1 - p_fraud), in [0.5, 1.0]. Below review_threshold the decision is
        "review" regardless of which side of 0.5 the score fell on."""
        p_fraud = self.predict_proba_fraud(X)
        confidence = np.maximum(p_fraud, 1 - p_fraud)
        decision = np.where(p_fraud >= 0.5, FRAUD, LEGIT).astype(object)
        decision[confidence < review_threshold] = REVIEW
        return decision, p_fraud, confidence

    def coverage_precision_curve(
        self, X: ArrayLike, y: ArrayLike, thresholds: Sequence[float]
    ) -> list[dict[str, Any]]:
        """Sweep the review threshold over this model's own scores. The math
        lives in curve_from_scores (below) so calibrated scores can drive the
        exact same table - see calibration.py."""
        return curve_from_scores(self.predict_proba_fraud(X), y, thresholds)


def curve_from_scores(
    p_fraud: ArrayLike, y: ArrayLike, thresholds: Sequence[float]
) -> list[dict[str, Any]]:
    """The reject-knob table for a given fraud-score vector. For each threshold
    we report BOTH sides a fraud desk asks about - precision (are the auto-flags
    right?) and recall (what fraction of real fraud do we actually catch?):

      coverage              fraction of rows the model decides on its own (not
                            sent to review).
      precision_on_flagged  of the rows it auto-flags as fraud, the fraction
                            that are truly fraud. None when nothing is flagged
                            (precision is undefined, not zero).
      fraud_caught_auto     true frauds the model auto-flags (covered & fraud).
      fraud_in_review_queue true frauds routed to a human (below the threshold,
                            so surfaced for review, not silently cleared).
      fraud_missed          true frauds auto-cleared as legit (covered & legit) -
                            the only frauds that actually slip through.
      recall_auto           fraud_caught_auto / all test frauds: the fraction of
                            fraud the automated path catches on its own.

    The three fraud_* counts partition every true fraud (caught + queued +
    missed = total), so recall and the review-queue load are both explicit."""
    y = np.asarray(y).astype(int)
    p_fraud = np.asarray(p_fraud, dtype=float)
    confidence = np.maximum(p_fraud, 1 - p_fraud)
    predicted_fraud = p_fraud >= 0.5
    is_fraud = y == 1
    total_fraud = int(is_fraud.sum())

    rows: list[dict[str, Any]] = []
    for t in thresholds:
        covered = confidence >= t
        flagged = covered & predicted_fraud
        n_flagged = int(flagged.sum())
        precision = float((y[flagged] == 1).mean()) if n_flagged else None
        fraud_caught_auto = int((is_fraud & flagged).sum())
        fraud_in_review_queue = int((is_fraud & ~covered).sum())
        fraud_missed = int((is_fraud & covered & ~predicted_fraud).sum())
        recall_auto = fraud_caught_auto / total_fraud if total_fraud else None
        rows.append(
            {
                "review_threshold": round(float(t), 4),
                "coverage": round(float(covered.mean()), 4),
                "n_sent_to_review": int((~covered).sum()),
                "n_flagged_fraud": n_flagged,
                "precision_on_flagged": round(precision, 4) if precision is not None else None,
                "fraud_caught_auto": fraud_caught_auto,
                "fraud_in_review_queue": fraud_in_review_queue,
                "fraud_missed": fraud_missed,
                "recall_auto": round(recall_auto, 4) if recall_auto is not None else None,
            }
        )
    return rows


def expected_cost_curve(
    curve: list[dict[str, Any]],
    cost_missed_fraud: float,
    cost_false_flag: float,
    cost_review: float,
) -> list[dict[str, Any]]:
    """Price each operating point of a reject-knob curve. Costs are REQUIRED
    arguments with no defaults on purpose: real fraud costs are business
    numbers this repo cannot know, so it never bakes any in. Any costs shown
    in the docs are labeled illustrative.

      cost_missed_fraud  a fraud auto-cleared as legit (chargeback, loss)
      cost_false_flag    a legit transaction auto-flagged (friction, support)
      cost_review        one case routed to the human review queue

    Frauds that land in the review queue are deliberately NOT charged
    cost_missed_fraud: they were surfaced, and the queue's cost is already
    counted per-case via cost_review. That is the whole argument for the knob."""
    priced = []
    for row in curve:
        false_flags = row["n_flagged_fraud"] - row["fraud_caught_auto"]
        total = (
            row["fraud_missed"] * cost_missed_fraud
            + false_flags * cost_false_flag
            + row["n_sent_to_review"] * cost_review
        )
        priced.append(
            {
                "review_threshold": row["review_threshold"],
                "n_false_flags": int(false_flags),
                "expected_cost": round(float(total), 2),
            }
        )
    return priced
