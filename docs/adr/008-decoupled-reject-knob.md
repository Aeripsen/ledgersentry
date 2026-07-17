# ADR 008: decouple the reject knob's two cuts

Status: accepted

## Context

The reject knob sweeps one threshold `t`. `curve_from_scores` computes
`confidence = max(p, 1-p)` and reviews anything below `t`, which for any `t > 0.5`
decomposes into two cuts:

```
p >= t        -> auto-flag fraud
1-t < p < t   -> review
p <= 1-t      -> auto-clear legit
```

One parameter, two cuts, **forced symmetric about 0.5**. Asking to flag at 0.99
silently also demands clearing at 0.01.

That symmetry is inherited from FlowSentry (ADR 003), where the knob was a top-1
confidence over many classes. There it was the natural shape: with `k` classes,
"how sure am I of my top pick" is one quantity and there is no second cut to set.
Binary fraud scoring at a 0.13% base rate is a different problem wearing the same
API, and the symmetry is wrong for it in a specific, measurable way: the two lanes
want to live at wildly different places on the scale. A desk wants to flag at
`p >= 0.9` and clear at `p <= 0.02`, and this API cannot express that operating
point at all. `(0.9, 0.02)` is not a threshold the knob has; the closest it can
say is `t = 0.9`, which means `(0.9, 0.1)`, or `t = 0.98`, which means
`(0.98, 0.02)` and throws the flag lane away to get the clear lane.

It is the root cause of both committed curves' degenerate high ends: the raw
curve's clear lane starves at t=0.99 and the calibrated curve's flag lane starves
at t=0.9, and neither is fixable by choosing better thresholds, because the defect
is that one number is choosing two things.

## Decision

Add `decoupled_curve_from_scores(p, y, [(flag_at, clear_at), ...])` beside the
existing sweep, and publish it in calibrated units as `decoupled_curve_calibrated`
in `artifacts/calibration_<source>.json`.

**Added, never substituted.** `curve_from_scores` is untouched, every committed
symmetric number is byte-identical, and the headline (`pr_auc` 0.7278,
`recall_at_full_coverage` 0.84) does not live in this file at all. The decoupled
function is a strict generalization: for any `t > 0.5`, `(t, 1-t)` reproduces
`curve_from_scores(t)` row for row, which `tests/test_model.py` pins across
t = 0.6 .. 1.0. `t = 0.5` is excluded because there the cuts collide at exactly
0.5 and the lanes would overlap.

**Calibrated units only, not raw.** "Flag at 0.9" means "flag at 90% likely
fraud" only after calibration; on raw scores the same literal is a position on an
unlabeled ladder (ADR 002, ADR 006). Sweeping decoupled points on raw scores would
invite exactly the misreading calibration exists to prevent, and `train.py`'s
committed curve stays as it is.

**Overlapping lanes are a loud error.** `flag_at > clear_at` is enforced with a
`ValueError`. A row that is both auto-flagged and auto-cleared is a contradiction,
not an operating point, and a silent precedence rule is the kind of thing nobody
remembers six months later.

## Consequences

The measured payoff on ULB, from `decoupled_curve_calibrated` beside
`coverage_precision_curve_calibrated` in the same artifact:

| operating point | flags | precision | caught / queue / missed |
|---|---|---|---|
| symmetric t=0.50, i.e. (0.50, 0.50) | 58 | 87.93% | 51 / 0 / **24** |
| symmetric t=0.99, i.e. (0.99, 0.01) | **0** | undefined | 0 / 60 / 15 |
| **decoupled (0.50, 0.01)** | **58** | **87.93%** | 51 / 9 / **15** |

The symmetric knob offers a choice: keep 58 flags and let 24 frauds through, or
cut the misses to 15 and lose every flag. The decoupled knob takes both halves at
once, for 146 reviews out of 56,961 rows. `fraud_missed` is a function of
`clear_at` alone, which is why the coupling was costing real money in the first
place, and why no threshold list could have recovered it.

It also reaches points the sweep could not: `(0.85, 0.01)` flags 50 at **94.00%**
precision, above anything on the symmetric calibrated curve.

**What it does not fix, stated because decoupling is not magic.** It removes the
forced link between the cuts; it does not invent scores the model never produces.
`(0.9, 0.02)` is kept in the committed sweep and flags **nothing**, because this
model's highest calibrated score on the fold is 0.856496 (`score_range_test.platt.max`).
The desk's canonical ask is now expressible and still unmet, which is a model
ceiling rather than an API defect. Keeping that row is the honest result.

## Rejected

- **Refactoring `curve_from_scores` to delegate to the decoupled version:**
  tempting, since the symmetric curve is a special case of it. But `t = 0.5` is
  genuinely a different case (coverage 1.0, no overlap possible), it is the row
  the headline `recall_at_full_coverage` reads, and a refactor that moves the
  headline is a regression however clean it looks. Two functions, one pinned to
  the other by test.
- **Replacing the symmetric sweep outright:** it is what every committed number
  and both published tables are built on, and the equivalence test needs a
  reference to check against.
- **Sweeping decoupled points on raw scores too:** above. The units would not mean
  what the parameter names say.
- **Auto-tuning `(flag_at, clear_at)` to optimize expected cost:** that requires
  the cost numbers this repo refuses to invent (ADR 003). The curve is published;
  the caller picks the point and brings their own costs.
- **A `clear_at = None` "flag-only" mode:** an abstraction with no second
  implementer today. `clear_at = 0.0` already expresses it.
