"""
Uncertainty on the headline, because one fold with 75 positives does not earn
four decimal places.

The gap this closes: train.py publishes PR-AUC to four decimals off a single
temporal fold whose entire positive class is 75 rows, and the model card tells
readers not to over-read differences between adjacent rows. Both statements are
true, and together they are an admission that the precision printed is not the
precision earned. This pipeline replaces the shrug with an interval.

It runs as its own pipeline beside train.py, for the same reason calibration
does: the committed metrics file is byte-identity checked (scripts/verify_repro.py),
and folding a resampling loop into it would tie that guarantee to an RNG for no
benefit. The headline numbers are not recomputed here, they are re-derived and
compared - if the point estimates below ever stop matching the committed
metrics_<source>.json, this interval describes some other model and is worthless.

Method: the percentile bootstrap. Resample the test fold with replacement
(n = the fold's own size), recompute the statistic on each resample, take the
2.5th and 97.5th percentiles of the resulting distribution. Seeded, so the
interval reproduces exactly like every other number in this repo.

Two honest limits, stated up front because an interval invites more trust than
it has earned:

  1. It captures SAMPLING noise only: "how much would this number move if those
     7.6 hours had held a slightly different draw of transactions?" It says
     nothing about FOLD-CHOICE variance. Pick a different 7.6 hours and the
     answer can move further than this interval suggests. Rolling-origin
     evaluation over multiple temporal folds is what measures that; it is an
     open gap on the README roadmap and a bootstrap cannot substitute for it.
  2. It resamples rows independently, which assumes the fold's rows are
     exchangeable. Fraud is bursty and campaign-driven, so real frauds arrive
     correlated in time. Independent resampling therefore probably UNDERSTATES
     the true uncertainty. A block bootstrap over time windows would respect
     that structure and is not implemented, so read these widths as a floor on
     the uncertainty, not the whole of it.

Why percentile and not BCa: BCa corrects for bias and skew and is the better
default in general, but its acceleration term needs a jackknife, one
recomputation per row, 56,961 of them on ULB, to buy a correction that is small
when B is large and the statistic is this smooth. Percentile is the standard,
defensible choice at this scale, and naming the trade-off is the point.

Run: python scripts/bootstrap.py   -> artifacts/bootstrap_<source>.json
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import average_precision_score

from .config import get_settings
from .data import build_preprocessor, engineer_time_features, load, temporal_grouped_split
from .model import FraudDetector

# 97.5th percentile of the standard normal, i.e. the z for a two-sided 95%
# interval. Hardcoded rather than imported from scipy.stats: scipy is only a
# TRANSITIVE dependency here (scikit-learn pulls it in) and requirements.txt
# does not pin it, so importing it directly would be an unpinned dependency
# smuggled into a repo whose whole claim is pinned reproduction.
Z_95 = 1.959964

CI_PERCENTILES = (2.5, 97.5)


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Wilson rather than the Wald/normal approximation on purpose: at small n and
    a proportion far from 0.5, Wald has poor coverage and can hand back limits
    outside [0, 1]. Wilson inverts the score test instead, stays in range, and
    is asymmetric - which is the correct shape when the estimate is 0.84 and n
    is 75.

    Recall is a proportion over the fold's frauds, so it has a closed-form
    interval that owes nothing to resampling. It is computed here as an
    INDEPENDENT check on the bootstrap: two unrelated methods should land in the
    same place, and if they ever diverge the bootstrap is the one to distrust.
    """
    if trials <= 0:
        return (float("nan"), float("nan"))
    p = successes / trials
    denom = 1.0 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denom
    half = (z / denom) * float(
        np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2))
    )
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


def _pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p))


def _recall_at_full_coverage(y: np.ndarray, p: np.ndarray) -> float:
    """The committed headline recall: fraction of true frauds the automated path
    catches at threshold 0.5, where nothing is sent to review. Mirrors
    curve_from_scores' recall_auto at the 0.5 row by construction."""
    is_fraud = y == 1
    total = int(is_fraud.sum())
    if total == 0:
        return float("nan")
    return float(((p >= 0.5) & is_fraud).sum() / total)


def _summarize(point: float, samples: np.ndarray) -> dict:
    lo, hi = (float(v) for v in np.percentile(samples, CI_PERCENTILES))
    return {
        "point_estimate": round(point, 4),
        "ci_lower": round(lo, 4),
        "ci_upper": round(hi, 4),
        "ci_width": round(hi - lo, 4),
        # the bootstrap distribution's own spread, reported beside the interval
        # so the interval is not the only summary of it
        "std_error": round(float(np.std(samples, ddof=1)), 4),
    }


def bootstrap_headline(
    y: np.ndarray, p: np.ndarray, n_resamples: int, seed: int
) -> dict:
    """Percentile-bootstrap PR-AUC and recall-at-full-coverage on one score
    vector. Returns the point estimates beside their intervals."""
    rng = np.random.default_rng(seed)
    n = len(y)
    pr_samples: list[float] = []
    recall_samples: list[float] = []
    degenerate = 0

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y_b, p_b = y[idx], p[idx]
        # A resample with zero frauds leaves both statistics undefined. At 75
        # positives in 56,961 rows this has probability ~e^-75 and never fires,
        # but it is counted rather than assumed away: on a smaller fold it could,
        # and a silently-dropped resample would bias the interval.
        if int((y_b == 1).sum()) == 0:
            degenerate += 1
            continue
        pr_samples.append(_pr_auc(y_b, p_b))
        recall_samples.append(_recall_at_full_coverage(y_b, p_b))

    return {
        "pr_auc": _summarize(_pr_auc(y, p), np.asarray(pr_samples)),
        "recall_at_full_coverage": _summarize(
            _recall_at_full_coverage(y, p), np.asarray(recall_samples)
        ),
        "n_degenerate_resamples": degenerate,
    }


def main() -> dict:
    cfg = get_settings()

    df = load(data_dir=cfg.data_dir)
    source = df.attrs.get("source", "unknown")
    df = engineer_time_features(df)
    # the identical split train.py uses, so these are the committed headline's
    # own predictions and not a lookalike
    train_df, test_df = temporal_grouped_split(df, test_size=cfg.test_size)

    pre = build_preprocessor(train_df)
    X_train = pre.fit_transform(train_df)
    X_test = pre.transform(test_df)
    y_train = train_df["is_fraud"].to_numpy()
    y_test = test_df["is_fraud"].to_numpy()

    model = FraudDetector(
        random_state=cfg.random_state,
        max_iter=cfg.max_iter,
        learning_rate=cfg.learning_rate,
        model=cfg.model,
    ).fit(X_train, y_train)
    p_test = model.predict_proba_fraud(X_test)

    n_test_fraud = int((y_test == 1).sum())
    print(
        f"[data ] source={source}: test={len(test_df)} ({n_test_fraud} fraud) "
        f"resamples={cfg.bootstrap_resamples} seed={cfg.random_state}"
    )

    stats = bootstrap_headline(
        y_test, p_test, n_resamples=cfg.bootstrap_resamples, seed=cfg.random_state
    )

    # Independent closed-form check on the recall interval (see wilson_interval).
    caught = int(((p_test >= 0.5) & (y_test == 1)).sum())
    w_lo, w_hi = wilson_interval(caught, n_test_fraud)

    # What one more caught fraud is worth, in points of recall. The blunt way to
    # say "these decimals are noise": if a single label moves the headline by
    # more than the difference you are arguing about, you are arguing about
    # nothing.
    one_fraud_pts = round(100.0 / n_test_fraud, 4) if n_test_fraud else None

    report = {
        "data_source": source,
        "is_synthetic": source == "synthetic",
        "model": cfg.model,
        "method": "percentile bootstrap, resampled with replacement over the test fold",
        "n_resamples": cfg.bootstrap_resamples,
        "seed": cfg.random_state,
        "confidence_level": 0.95,
        "n_test": int(len(test_df)),
        "n_test_fraud": n_test_fraud,
        **stats,
        "recall_wilson_95": {
            "successes": caught,
            "trials": n_test_fraud,
            "ci_lower": round(w_lo, 4),
            "ci_upper": round(w_hi, 4),
            "note": (
                "Closed-form Wilson score interval on the same proportion, computed "
                "independently of the bootstrap as a cross-check. Agreement is "
                "evidence the resampling is sane; divergence would mean distrust "
                "the bootstrap, not the Wilson."
            ),
        },
        "one_extra_fraud_moves_recall_pts": one_fraud_pts,
        "limitations": [
            "Sampling noise only. This does NOT capture fold-choice variance: a "
            "different temporal cut can move the number further than these bounds. "
            "Rolling-origin evaluation is the fix and is an open gap on the roadmap.",
            "Rows are resampled independently, which assumes exchangeability. Fraud "
            "is bursty and campaign-driven, so real positives are time-correlated "
            "and this interval likely understates the truth. Read it as a floor.",
            "Percentile method, not BCa: no bias or skew correction (BCa's "
            "acceleration term needs a 56,961-point jackknife on ULB).",
        ],
    }

    pr, rc = report["pr_auc"], report["recall_at_full_coverage"]
    print(
        f"[boot ] PR-AUC {pr['point_estimate']} "
        f"95% CI [{pr['ci_lower']}, {pr['ci_upper']}] (width {pr['ci_width']})"
    )
    print(
        f"[boot ] recall {rc['point_estimate']} "
        f"95% CI [{rc['ci_lower']}, {rc['ci_upper']}] (width {rc['ci_width']})"
    )
    print(f"[check] recall Wilson 95% CI [{round(w_lo, 4)}, {round(w_hi, 4)}] "
          f"({caught}/{n_test_fraud}) - independent of the bootstrap")
    print(f"[check] one more caught fraud moves recall {one_fraud_pts} points")

    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.artifact_dir / f"bootstrap_{source}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[save ] {out}")
    return report


if __name__ == "__main__":
    main()
