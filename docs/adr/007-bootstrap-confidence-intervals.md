# ADR 007: confidence intervals on the headline, as their own pipeline

Status: accepted

## Context

The headline is PR-AUC 0.7278 and recall 0.84, published to four decimals off a
single temporal fold whose entire positive class is **75 rows**. The model card
told readers not to over-read differences between adjacent rows, which was true
and was also an admission: if the decimals cannot be read, they should not have
been printed without a number saying how far they can move.

This is the first thing a statistically literate reader attacks, and it is the
cheapest credibility available to a repo whose whole pitch is that its numbers
are true. "75 positives, is that meaningful?" deserves arithmetic, not a shrug.

## Decision

A separate pipeline, `scripts/bootstrap.py` -> `artifacts/bootstrap_<source>.json`,
publishes 95% intervals on PR-AUC and recall-at-full-coverage, and the model card
and README print the interval **beside the point estimate** rather than in a
footnote.

Three sub-decisions, each with a reason:

**Its own pipeline, not folded into `train.py`.** Same argument as ADR 006. The
committed metrics file is byte-identity checked by `scripts/verify_repro.py`, and
folding a resampling loop into it would tie that guarantee to an RNG for no
benefit. The bootstrap re-derives the headline predictions through the identical
split and refit, so its `point_estimate` fields must equal the committed
`metrics_<source>.json` values. They do (0.7278 / 0.84). If they ever stop, the
interval describes some other model and is worthless.

**Percentile bootstrap, not BCa.** BCa corrects for bias and skew and is the
better default in general, but its acceleration term needs a jackknife - one
recomputation per row, 56,961 of them on ULB - to buy a correction that is small
when B is large and the statistic is this smooth. Percentile is the standard,
defensible choice at this scale.

**A closed-form Wilson interval beside the resampled one.** Recall is a binomial
proportion over the fold's 75 frauds, so it has an interval that owes nothing to
resampling. Computing both is a free cross-check on the bootstrap by a method that
shares none of its machinery: bootstrap [0.75, 0.9167] against Wilson
[0.7408, 0.9060], agreeing to about a point at each end. Wilson rather than Wald
because at n=75 and p=0.84 the normal approximation has poor coverage and can
produce limits above 1.0; Wilson inverts the score test, stays in range, and is
asymmetric, which is correct off 0.5.

## Consequences

- **The headline now carries a 0.20-wide interval, in public.** PR-AUC
  [0.6214, 0.8232]. That is not a flattering number and it is the point: the
  conclusion "far better than no-skill" survives it (the floor is still ~480x the
  0.0013 baseline), and any argument resting on the decimals does not.
- Claims elsewhere in the repo are now checkable against it, and one gets weaker:
  the calibration model's 0.7544 sits inside this interval, so 0.7278 vs 0.7544 is
  **not** a demonstrated difference and the model card no longer implies it is.
- The interval measures **sampling noise only**. It does not capture fold-choice
  variance, which needs rolling-origin evaluation across multiple temporal folds
  and remains an open gap. An interval invites more trust than it has earned, so
  that limit is committed in the artifact's own `limitations` field rather than
  living only in prose.
- Rows are resampled independently, which assumes exchangeability. Fraud is bursty
  and campaign-driven, so real positives are time-correlated and these widths are a
  **floor** on the uncertainty. A block bootstrap would respect that structure; it
  is not built, and saying so is cheaper than being caught by it.

## Rejected

- **Reporting the bootstrap mean as the headline:** the point estimate is the
  statistic on the real fold. Publishing a resampled mean would quietly ship a
  different number than the committed metrics, and byte-identity would be a lie.
- **A CI inside `metrics_<source>.json`:** couples the byte-identity guarantee to
  an RNG. Separate artifact, same as calibration and the benchmark.
- **Dropping resamples with zero positives silently:** at 75 positives this has
  probability ~e^-75 and never fires, but a silently-discarded resample biases an
  interval. They are counted (`n_degenerate_resamples`) instead.
- **Widening the thresholds sweep or rounding to fewer decimals instead:** that
  hides the problem rather than measuring it. The four decimals stay, with the
  interval next to them, because the artifact is what it is.
- **scipy for the normal quantile:** scipy is only a transitive dependency here
  (via scikit-learn) and `requirements.txt` does not pin it. Importing it directly
  would smuggle an unpinned dependency into a repo whose claim is pinned
  reproduction. The constant is hardcoded and unit-tested against `math.erf`.
