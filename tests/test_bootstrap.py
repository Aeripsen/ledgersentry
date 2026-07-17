"""
An interval is a claim like any other, so it gets the same treatment: the
closed-form one is pinned to a hand-computable value, the resampled one has to
be deterministic, and the two have to agree with each other on the statistic
they both estimate.
"""
import numpy as np
import pytest

from ledgersentry.bootstrap import (
    Z_95,
    bootstrap_headline,
    wilson_interval,
)


def _scores_and_labels(n=2000, seed=3):
    """Separable-ish scores with a small positive class, the shape this repo
    actually evaluates on."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    p = np.clip(rng.normal(0.2, 0.1, size=n) + 0.5 * y, 0.001, 0.999)
    return p, y


def test_wilson_matches_hand_computation():
    """The committed headline's own interval, checkable on paper: recall 63/75.

    Wilson: center = (p + z^2/2n) / (1 + z^2/n),
            half   = z/(1 + z^2/n) * sqrt(p(1-p)/n + z^2/4n^2)
    with p = 0.84, n = 75, z = 1.959964 this gives [0.7408, 0.9060].
    """
    lo, hi = wilson_interval(63, 75)
    assert lo == pytest.approx(0.7408, abs=1e-4)
    assert hi == pytest.approx(0.9060, abs=1e-4)


def test_wilson_stays_in_range_at_the_boundary():
    """Why Wilson and not Wald: at p=1.0 the normal approximation puts the upper
    limit above 1.0 and the interval has zero width. Wilson must do neither."""
    lo, hi = wilson_interval(75, 75)
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0
    assert lo < 1.0  # a real interval, not a degenerate point


def test_wilson_is_asymmetric_about_the_estimate():
    """At an estimate far from 0.5 the interval must lean, which is the whole
    reason for preferring it to a symmetric Wald interval here."""
    lo, hi = wilson_interval(63, 75)
    point = 63 / 75
    assert (point - lo) > (hi - point)


def test_wilson_degenerate_trials_is_nan_not_a_crash():
    lo, hi = wilson_interval(0, 0)
    assert np.isnan(lo) and np.isnan(hi)


def test_z_95_is_the_right_quantile():
    """Guard the hardcoded constant: 1.959964 must be the standard normal's
    97.5th percentile. Checked against the error function so the test does not
    just restate the literal."""
    from math import erf, sqrt

    two_sided_mass = erf(Z_95 / sqrt(2.0))
    assert two_sided_mass == pytest.approx(0.95, abs=1e-6)


def test_bootstrap_is_deterministic():
    """Same seed, same interval - the committed numbers are worthless if a
    rerun moves them."""
    p, y = _scores_and_labels()
    a = bootstrap_headline(y, p, n_resamples=50, seed=42)
    b = bootstrap_headline(y, p, n_resamples=50, seed=42)
    assert a == b


def test_bootstrap_different_seed_moves_the_interval():
    """Sanity: the resampling is actually resampling, not returning a constant."""
    p, y = _scores_and_labels()
    a = bootstrap_headline(y, p, n_resamples=50, seed=1)
    b = bootstrap_headline(y, p, n_resamples=50, seed=2)
    assert a["pr_auc"]["ci_lower"] != b["pr_auc"]["ci_lower"]


def test_bootstrap_interval_brackets_the_point_estimate():
    p, y = _scores_and_labels()
    out = bootstrap_headline(y, p, n_resamples=200, seed=42)
    for key in ("pr_auc", "recall_at_full_coverage"):
        s = out[key]
        assert s["ci_lower"] <= s["point_estimate"] <= s["ci_upper"]
        assert s["ci_width"] > 0


def test_bootstrap_point_estimate_is_the_unresampled_statistic():
    """The point estimate must be the real number, never the bootstrap mean:
    reporting the resampled mean would quietly publish a different figure than
    the committed headline."""
    from sklearn.metrics import average_precision_score

    p, y = _scores_and_labels()
    out = bootstrap_headline(y, p, n_resamples=50, seed=42)
    assert out["pr_auc"]["point_estimate"] == round(
        float(average_precision_score(y, p)), 4
    )
    caught = int(((p >= 0.5) & (y == 1)).sum())
    assert out["recall_at_full_coverage"]["point_estimate"] == round(
        caught / int((y == 1).sum()), 4
    )


def test_bootstrap_and_wilson_agree_on_recall():
    """Two independent methods, one proportion. If these ever disagree
    materially, the resampling is wrong - that is what makes the cross-check
    worth committing."""
    p, y = _scores_and_labels()
    out = bootstrap_headline(y, p, n_resamples=400, seed=42)
    caught = int(((p >= 0.5) & (y == 1)).sum())
    w_lo, w_hi = wilson_interval(caught, int((y == 1).sum()))
    boot = out["recall_at_full_coverage"]
    assert boot["ci_lower"] == pytest.approx(w_lo, abs=0.06)
    assert boot["ci_upper"] == pytest.approx(w_hi, abs=0.06)


def test_bootstrap_counts_degenerate_resamples_instead_of_hiding_them():
    """With one positive, some resamples contain zero and both statistics are
    undefined there. They must be counted, not silently dropped."""
    rng = np.random.default_rng(0)
    y = np.zeros(40, dtype=int)
    y[0] = 1
    p = rng.random(40)
    out = bootstrap_headline(y, p, n_resamples=100, seed=7)
    assert out["n_degenerate_resamples"] > 0
