"""
Confidence calibration, done without touching the headline model.

The problem it fixes (measured, see docs/model_card.md): the raw model's
confidence is not a probability. On real ULB data its confidence almost never
exceeds 0.99, so a review threshold of 0.99 sends essentially the whole stream
to review - the knob's units are meaningless above 0.95. A calibrated
probability is what lets a fraud desk set the threshold by expected cost
(model.expected_cost_curve) instead of by reading a per-dataset curve.

What it does NOT fix, stated here because the committed numbers show it: the
knob has two cuts, `p >= t` to flag and `p <= 1-t` to clear, and calibration
helps one at the other's expense. Squashing the scale toward the base rate
lifts the clear lane (raw coverage at t=0.99 is 0.0003; calibrated it is
0.9964) and starves the flag lane, because the highest calibrated score on the
fold is 0.856496 (`score_range_test.platt`) so nothing can be flagged at all
above t=0.85. Calibration MOVED this curve's degenerate end from 0.99 to 0.9,
it did not remove it. Both curves are committed in full either way.

The design constraint that shapes everything here: a calibrator must be fit on
data its model was NOT trained on, or it just certifies the model's own
overconfidence. Carving that data out of the train window necessarily costs
training rows, which would move the committed headline numbers. So calibration
runs as its own pipeline beside train.py, never inside it:

    fit slice   = first (1 - calibration_size) of the TRAIN window, temporally
    cal slice   = the rest of the TRAIN window (after the fit slice in time,
                  same leakage-safe split machinery as everything else)
    test fold   = the SAME held-out test fold train.py evaluates on, untouched

Two calibrators are fit on the cal slice and reported side by side - isotonic
regression (non-parametric) and Platt scaling (a sigmoid on the log-odds,
2 parameters). The SHIPPED choice is structural, decided before any numbers:
a calibrator here exists to fix the probability SCALE and must provably never
touch the ranking, so it must be strictly monotone - which Platt is and
isotonic is not (its step function maps ranges of distinct scores to one tied
value). Ties are not a technicality: with few positives to pin the steps,
isotonic collapses real score distinctions, and no cal-slice check can catch
that - isotonic evaluated on its own fit data looks fine by construction. The
ULB run shows exactly this (numbers in the committed
artifacts/calibration_ulb_creditcard.json): isotonic scores 0.7585 cal-slice
PR-AUC on its own data, then drops the untouched test fold's PR-AUC to 0.6728
vs 0.7544 raw, while Platt's test PR-AUC is bit-identical to raw and its
Brier improvement is just as large. Both calibrators' full cal and test
numbers are always written, win or lose, so the choice stays auditable.

Run: python scripts/calibrate.py   -> artifacts/calibration_<source>.json
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from .config import get_settings
from .data import build_preprocessor, engineer_time_features, load, temporal_grouped_split
from .model import FraudDetector, curve_from_scores, decoupled_curve_from_scores

# Logit clip bound. Deliberately near float64 resolution: any wider (say 1e-6)
# and distinct raw scores beyond the clip collapse into ties, which is exactly
# the ranking damage the strict-monotonicity rule exists to prevent - measured
# as a real PR-AUC change on the synthetic fixture before tightening.
_EPS = 1e-12
RELIABILITY_BINS = 10


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """Classic Platt scaling: a 2-parameter sigmoid fit on the model's
    log-odds. Effectively unregularized (large C), the standard choice -
    regularizing a 2-parameter map on thousands of points buys nothing."""

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        self._lr = LogisticRegression(C=1e10, max_iter=1000)
        self._lr.fit(_logit(np.asarray(p_raw))[:, None], np.asarray(y).astype(int))
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(_logit(np.asarray(p_raw))[:, None])[:, 1]

    @property
    def coefficients(self) -> tuple[float, float]:
        """(a, b) of the fitted map `p_cal = sigmoid(a * logit(p_raw) + b)`.
        Serialized into the artifact so the map itself is auditable, not just
        its outputs: with a and b you can invert it and check by hand what raw
        score any calibrated threshold demands."""
        return float(self._lr.coef_[0][0]), float(self._lr.intercept_[0])


class IsotonicCalibrator:
    """Non-parametric monotone map. More flexible than Platt, but needs enough
    positives in the cal slice to be trustworthy - with few frauds it can
    produce coarse steps. That trade-off is exactly why both are fit and
    compared instead of hardcoding one."""

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
        self._iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self._iso.fit(np.asarray(p_raw), np.asarray(y).astype(int))
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        return self._iso.transform(np.asarray(p_raw))


def _score_range(p: np.ndarray) -> dict:
    """Observed floor/ceiling of a score vector. Small, but load-bearing: the
    reject knob's usable range is bounded by these two numbers, not by the
    threshold list we happen to sweep."""
    return {"min": round(float(np.min(p)), 6), "max": round(float(np.max(p)), 6)}


def _reliability(y: np.ndarray, p: np.ndarray) -> dict:
    """Quantile-binned reliability curve: mean predicted probability vs
    observed fraud rate per bin. Quantile bins because at a 0.1% base rate
    uniform bins would put every transaction in the first bin."""
    prob_true, prob_pred = calibration_curve(
        y, p, n_bins=RELIABILITY_BINS, strategy="quantile"
    )
    return {
        "strategy": "quantile",
        "n_bins": len(prob_true),
        "mean_predicted": [round(float(v), 6) for v in prob_pred],
        "observed_fraud_rate": [round(float(v), 6) for v in prob_true],
    }


def main() -> dict:
    cfg = get_settings()

    df = load(data_dir=cfg.data_dir)
    source = df.attrs.get("source", "unknown")
    df = engineer_time_features(df)
    # identical split to train.py, so the test fold is the same rows the
    # committed headline metrics are measured on
    train_df, test_df = temporal_grouped_split(df, test_size=cfg.test_size)
    fit_df, cal_df = temporal_grouped_split(train_df, test_size=cfg.calibration_size)
    print(
        f"[data ] source={source}: fit={len(fit_df)} ({int(fit_df['is_fraud'].sum())} fraud) "
        f"cal={len(cal_df)} ({int(cal_df['is_fraud'].sum())} fraud) "
        f"test={len(test_df)} ({int(test_df['is_fraud'].sum())} fraud)"
    )

    pre = build_preprocessor(fit_df)
    X_fit = pre.fit_transform(fit_df)
    X_cal = pre.transform(cal_df)
    X_test = pre.transform(test_df)
    y_fit = fit_df["is_fraud"].to_numpy()
    y_cal = cal_df["is_fraud"].to_numpy()
    y_test = test_df["is_fraud"].to_numpy()

    model = FraudDetector(
        random_state=cfg.random_state,
        max_iter=cfg.max_iter,
        learning_rate=cfg.learning_rate,
        model=cfg.model,
    ).fit(X_fit, y_fit)

    p_cal = model.predict_proba_fraud(X_cal)
    p_test = model.predict_proba_fraud(X_test)

    # the honest cost of carving out the cal slice: this model saw less data
    # than the headline one, so report its own PR-AUC beside the headline
    pr_auc_calibration_model = float(average_precision_score(y_test, p_test))

    platt = PlattCalibrator().fit(p_cal, y_cal)
    calibrators: dict[str, IsotonicCalibrator | PlattCalibrator] = {
        "isotonic": IsotonicCalibrator().fit(p_cal, y_cal),
        "platt": platt,
    }
    brier_cal_slice = {
        name: float(brier_score_loss(y_cal, c.transform(p_cal)))
        for name, c in calibrators.items()
    }
    pr_auc_cal_raw = float(average_precision_score(y_cal, p_cal))
    pr_auc_cal_slice = {
        name: float(average_precision_score(y_cal, c.transform(p_cal)))
        for name, c in calibrators.items()
    }
    # structural choice, not an empirical race - see module docstring: the
    # shipped calibrator must be strictly monotone, and only Platt is.
    chosen = "platt"

    brier_test = {"raw": float(brier_score_loss(y_test, p_test))}
    pr_auc_test = {"raw": pr_auc_calibration_model}
    # The observed score FLOOR and CEILING on the test fold, per scale. These are
    # what actually decide where the reject knob dies, so they are committed
    # rather than left to a re-run: the knob's two cuts are `p >= t` (flag) and
    # `p <= 1-t` (clear), so any t above the ceiling can flag nothing, and any
    # 1-t below the floor can clear nothing. The raw ceiling is why the raw
    # curve's clear lane starves at t=0.99, and the calibrated ceiling is why
    # its flag lane starves at t=0.9. See docs/model_card.md, "What calibration
    # does NOT fix".
    score_range_test = {"raw": _score_range(p_test)}
    for name, c in calibrators.items():
        p = c.transform(p_test)
        brier_test[name] = float(brier_score_loss(y_test, p))
        # a monotone map cannot improve ranking; reported to PROVE it did not hurt
        pr_auc_test[name] = float(average_precision_score(y_test, p))
        score_range_test[name] = _score_range(p)

    p_chosen = calibrators[chosen].transform(p_test)
    report = {
        "data_source": source,
        "is_synthetic": source == "synthetic",
        "model": cfg.model,
        "calibration_size": cfg.calibration_size,
        "n_fit": int(len(fit_df)),
        "n_cal": int(len(cal_df)),
        "n_cal_fraud": int(y_cal.sum()),
        "n_test": int(len(test_df)),
        "n_test_fraud": int(y_test.sum()),
        "chosen_calibrator": chosen,
        "chosen_on": (
            "structural rule, decided before fitting: the shipped calibrator "
            "must be strictly monotone (provably zero ranking damage); "
            "isotonic is fit and fully reported beside it for comparison"
        ),
        "pr_auc_cal_slice": {
            "raw": round(pr_auc_cal_raw, 4),
            **{k: round(v, 4) for k, v in pr_auc_cal_slice.items()},
        },
        "brier_cal_slice": {k: round(v, 6) for k, v in brier_cal_slice.items()},
        "brier_test": {k: round(v, 6) for k, v in brier_test.items()},
        "pr_auc_test": {k: round(v, 4) for k, v in pr_auc_test.items()},
        "score_range_test": score_range_test,
        "platt_map": {
            "form": "p_cal = sigmoid(a * logit(p_raw) + b)",
            "a": round(platt.coefficients[0], 6),
            "b": round(platt.coefficients[1], 6),
        },
        "reliability_raw": _reliability(y_test, p_test),
        "reliability_calibrated": _reliability(y_test, p_chosen),
        # the reject knob in CALIBRATED probability units - the point of it all
        "coverage_precision_curve_calibrated": curve_from_scores(
            p_chosen, y_test, cfg.review_thresholds
        ),
        # the same knob with its two cuts DECOUPLED (ADR 008). The curve above
        # forces clear_at = 1 - flag_at, which is why its flag lane dies at 0.9;
        # here flag_at and clear_at are independent, so a desk can hold a strict
        # flag bar and a generous clear bar at once. Added beside the symmetric
        # curve, never replacing it: every committed number above is unchanged.
        "decoupled_curve_calibrated": decoupled_curve_from_scores(
            p_chosen, y_test, cfg.decoupled_operating_points
        ),
    }

    print(f"[cal  ] cal-slice pr_auc raw={pr_auc_cal_raw:.4f} {pr_auc_cal_slice}")
    print(f"[cal  ] chosen={chosen} by the strict-monotonicity rule "
          f"(cal-slice brier {brier_cal_slice})")
    print(f"[test ] brier: {brier_test}")
    print(f"[test ] pr_auc: {pr_auc_test}")

    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.artifact_dir / f"calibration_{source}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[save ] {out}")
    return report


if __name__ == "__main__":
    main()
