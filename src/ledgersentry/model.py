"""
Baseline fraud classifier with a tunable reject (abstain -> "review") knob.

Model: `HistGradientBoostingClassifier` (gradient-boosted trees, scikit-learn).
Chosen over xgboost because it ships inside scikit-learn - already a dependency,
so it installs cleanly everywhere including CI with no compiled-wheel risk - and
it natively handles missing values in numeric features, which matters once a
real dataset with sparse anonymized columns (ULB's V1..V28, IEEE-CIS's C/D/V
columns) is dropped in. xgboost is a documented drop-in alternative if a future
run wants it.

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

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier

FRAUD = "fraud"
LEGIT = "legit"
REVIEW = "review"


class FraudDetector(BaseEstimator):
    def __init__(self, random_state: int = 42, max_iter: int = 200, learning_rate: float = 0.1):
        self.random_state = random_state
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def fit(self, X, y) -> FraudDetector:
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)

        # Balanced sample weights, same formula as sklearn's class_weight="balanced"
        # (n_samples / (n_classes * count_per_class)), computed from TRAIN labels
        # only. Passed as sample_weight so it works across sklearn versions without
        # depending on HistGradientBoostingClassifier's own class_weight support.
        counts = np.bincount(y, minlength=int(y.max()) + 1)
        weight_per_class = counts.sum() / (len(counts) * np.maximum(counts, 1))
        sample_weight = weight_per_class[y]

        self.model_ = HistGradientBoostingClassifier(
            random_state=self.random_state,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
        )
        self.model_.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba_fraud(self, X) -> np.ndarray:
        """P(fraud) per row - the positive-class column of predict_proba."""
        proba = self.model_.predict_proba(X)
        fraud_col = list(self.model_.classes_).index(1)
        return proba[:, fraud_col]

    def decide(self, X, review_threshold: float = 0.0):
        """Return (decision, p_fraud, confidence).
        decision[i] in {"fraud", "legit", "review"}. confidence = max(p_fraud,
        1 - p_fraud), in [0.5, 1.0]. Below review_threshold the decision is
        "review" regardless of which side of 0.5 the score fell on."""
        p_fraud = self.predict_proba_fraud(X)
        confidence = np.maximum(p_fraud, 1 - p_fraud)
        decision = np.where(p_fraud >= 0.5, FRAUD, LEGIT).astype(object)
        decision[confidence < review_threshold] = REVIEW
        return decision, p_fraud, confidence

    def coverage_precision_curve(self, X, y, thresholds) -> list[dict]:
        """Sweep the review threshold. For each threshold we report BOTH sides a
        fraud desk asks about - precision (are the auto-flags right?) and recall
        (what fraction of real fraud do we actually catch?):

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
        p_fraud = self.predict_proba_fraud(X)
        confidence = np.maximum(p_fraud, 1 - p_fraud)
        predicted_fraud = p_fraud >= 0.5
        is_fraud = y == 1
        total_fraud = int(is_fraud.sum())

        rows = []
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
