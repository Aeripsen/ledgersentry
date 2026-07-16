"""
Confidence calibration, done without touching the headline model.

The problem it fixes (measured, see docs/model_card.md): the raw model's
confidence is not a probability. On real ULB data its confidence almost never
exceeds 0.99, so a review threshold of 0.99 sends essentially the whole stream
to review - the knob's units are meaningless above 0.95. A calibrated
probability is what lets a fraud desk set the threshold by expected cost
(model.expected_cost_curve) instead of by reading a per-dataset curve.

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
from .model import FraudDetector, curve_from_scores

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

    calibrators: dict[str, IsotonicCalibrator | PlattCalibrator] = {
        "isotonic": IsotonicCalibrator().fit(p_cal, y_cal),
        "platt": PlattCalibrator().fit(p_cal, y_cal),
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
    for name, c in calibrators.items():
        p = c.transform(p_test)
        brier_test[name] = float(brier_score_loss(y_test, p))
        # a monotone map cannot improve ranking; reported to PROVE it did not hurt
        pr_auc_test[name] = float(average_precision_score(y_test, p))

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
        "reliability_raw": _reliability(y_test, p_test),
        "reliability_calibrated": _reliability(y_test, p_chosen),
        # the reject knob in CALIBRATED probability units - the point of it all
        "coverage_precision_curve_calibrated": curve_from_scores(
            p_chosen, y_test, cfg.review_thresholds
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
