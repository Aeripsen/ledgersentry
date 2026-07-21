"""
Expected cost as a function of the review threshold, as a committed artifact.

The model card used to quote cost figures with no artifact behind them, and
those numbers were deleted (git history: "drop unsourced expected-cost
figures") because a dollar figure with nothing reproducible under it is exactly
the kind of claim this repo exists not to make. `model.expected_cost_curve` has
always been able to compute the curve, but nothing ran it and committed the
output. This pipeline does, which turns the removed claim into a measured one.

What it prices: the calibrated reject knob. Cost per threshold needs
probabilities, not raw confidence scores, because you multiply a probability by
a dollar amount and you cannot multiply a ranking score by one. So the curve is
computed on Platt-calibrated test-fold scores (the same calibrator calibration.py
ships), on the same held-out fold the headline is measured on.

Costs stay illustrative and stay explicit. Real fraud costs are business numbers
this repo cannot know, so `expected_cost_curve` takes them as required arguments
with no defaults, and this pipeline prices SEVERAL made-up triples rather than
one, precisely so the artifact shows the optimum threshold MOVING with the
assumptions instead of implying a single true answer. Every number here is
`<count from the measured fold> x <a cost you supplied>`. Change the costs and
rerun; nothing is baked in.

The one real, cost-free fact in the output is `min_cost_threshold` per triple:
given those costs, which review threshold on this fold minimizes total expected
cost. That the optimum is not always full automation is the argument for having
a reject knob at all, and it is now shown with an artifact instead of asserted.

Run: python scripts/cost.py   -> artifacts/expected_cost_curve_<source>.json
"""
from __future__ import annotations

import json
import platform
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import sklearn

from .calibration import PlattCalibrator
from .config import get_settings
from .data import build_preprocessor, engineer_time_features, load, temporal_grouped_split
from .model import FraudDetector, curve_from_scores, expected_cost_curve

# A fine threshold grid so the curve reads as a curve, not six points. These are
# confidence thresholds in calibrated units: 0.5 is full automation (nothing
# sent to review), higher values route more of the uncertain middle to humans.
THRESHOLD_GRID: tuple[float, ...] = tuple(round(0.5 + 0.02 * i, 2) for i in range(25))

# Illustrative cost triples, made-up round numbers to show the mechanics. Named
# so the artifact says what each one assumes about the business, and spread so
# the optimum threshold lands in different places. (missed fraud, false flag,
# one human review), in the same dollars.
COST_SCENARIOS: tuple[dict, ...] = (
    {
        "name": "high_fraud_loss",
        "note": "a missed fraud is expensive relative to review; favors surfacing more",
        "cost_missed_fraud": 500.0,
        "cost_false_flag": 5.0,
        "cost_review": 2.0,
    },
    {
        "name": "balanced",
        "note": "the model card's old illustrative triple",
        "cost_missed_fraud": 200.0,
        "cost_false_flag": 5.0,
        "cost_review": 2.0,
    },
    {
        "name": "expensive_review",
        "note": "human review costs almost as much as eating a small fraud; favors automation",
        "cost_missed_fraud": 100.0,
        "cost_false_flag": 5.0,
        "cost_review": 20.0,
    },
)


def calibrated_test_scores(cfg) -> tuple[np.ndarray, np.ndarray]:
    """(y_test, calibrated p_fraud) on the same fold as the headline, using the
    same fit/cal/test carve-up as calibration.py. Recomputed here rather than
    imported so this pipeline stands alone like bootstrap and compare do."""
    df = engineer_time_features(load(data_dir=cfg.data_dir))
    train_df, test_df = temporal_grouped_split(df, test_size=cfg.test_size)
    fit_df, cal_df = temporal_grouped_split(train_df, test_size=cfg.calibration_size)

    pre = build_preprocessor(fit_df)
    X_fit = pre.fit_transform(fit_df)
    X_cal = pre.transform(cal_df)
    X_test = pre.transform(test_df)

    model = FraudDetector(
        random_state=cfg.random_state,
        max_iter=cfg.max_iter,
        learning_rate=cfg.learning_rate,
        model=cfg.model,
    ).fit(X_fit, fit_df["is_fraud"].to_numpy())

    platt = PlattCalibrator().fit(
        model.predict_proba_fraud(X_cal), cal_df["is_fraud"].to_numpy()
    )
    p_test = platt.transform(model.predict_proba_fraud(X_test))
    return test_df["is_fraud"].to_numpy(), p_test


def price_scenarios(
    curve: list[dict], scenarios: tuple[dict, ...]
) -> list[dict]:
    priced = []
    for s in scenarios:
        rows = expected_cost_curve(
            curve,
            cost_missed_fraud=s["cost_missed_fraud"],
            cost_false_flag=s["cost_false_flag"],
            cost_review=s["cost_review"],
        )
        best = min(rows, key=lambda r: r["expected_cost"])
        full_auto = next(r for r in rows if r["review_threshold"] == 0.5)
        priced.append(
            {
                **{k: s[k] for k in ("name", "note", "cost_missed_fraud",
                                     "cost_false_flag", "cost_review")},
                "min_cost_threshold": best["review_threshold"],
                "min_expected_cost": best["expected_cost"],
                "full_automation_cost": full_auto["expected_cost"],
                "savings_vs_full_automation": round(
                    full_auto["expected_cost"] - best["expected_cost"], 2
                ),
                "curve": rows,
            }
        )
    return priced


def main() -> dict:
    cfg = get_settings()
    source = load(data_dir=cfg.data_dir).attrs.get("source", "unknown")

    y_test, p_test = calibrated_test_scores(cfg)
    n_fraud = int((y_test == 1).sum())
    print(
        f"[data ] source={source}: test={len(y_test)} ({n_fraud} fraud), "
        f"calibrated scores, {len(THRESHOLD_GRID)} thresholds x "
        f"{len(COST_SCENARIOS)} illustrative cost triples"
    )

    knob = curve_from_scores(p_test, y_test, THRESHOLD_GRID)
    scenarios = price_scenarios(knob, COST_SCENARIOS)

    for s in scenarios:
        print(
            f"[cost ] {s['name']:16s} min at t={s['min_cost_threshold']} "
            f"cost {s['min_expected_cost']} vs {s['full_automation_cost']} full-auto "
            f"(saves {s['savings_vs_full_automation']})"
        )

    report = {
        "data_source": source,
        "is_synthetic": source == "synthetic",
        "measured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "score_units": "platt-calibrated P(fraud), same fold and calibrator as calibration.py",
        "n_test": int(len(y_test)),
        "n_test_fraud": n_fraud,
        "threshold_grid": list(THRESHOLD_GRID),
        "costs_are": (
            "ILLUSTRATIVE. Made-up round numbers to show the mechanics, not "
            "industry figures. expected_cost_curve takes costs as required "
            "arguments and ships none; change these and rerun. Every expected_cost "
            "is a measured count on this fold multiplied by a supplied cost."
        ),
        "counts_are_from": (
            "curve_from_scores on the calibrated test fold: n_flagged_fraud, "
            "fraud_caught_auto, fraud_missed, n_sent_to_review per threshold. "
            "false_flags = n_flagged_fraud - fraud_caught_auto. Frauds routed to "
            "the review queue are NOT charged as missed; that is the knob's whole "
            "argument (model.expected_cost_curve)."
        ),
        "scenarios": scenarios,
    }

    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.artifact_dir / f"expected_cost_curve_{source}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[save ] {out}")
    return report


if __name__ == "__main__":
    main()
