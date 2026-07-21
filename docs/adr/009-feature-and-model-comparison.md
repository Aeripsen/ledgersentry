# ADR 009: comparing feature sets and boosting configs without cheating the holdout

Status: accepted

## Context

Two things in this repo were asserted rather than shown. The feature set was
"whatever the source provides plus two time features", with no velocity family at
all, even though rolling counts are the first thing anyone doing card fraud
reaches for. And the boosting config behind the headline (`hist_gbdt`, 200 rounds,
learning rate 0.1) was a reasonable starting guess that became the committed
0.7278 by never being compared to anything.

Turning both into measured choices is easy to do badly. The dangerous version
fits eight variants, reads their test PR-AUCs, picks the best, and reports it. On
a fold with 75 positives that procedure will "find" a couple of points of
improvement out of pure noise every time, and the holdout that the whole repo's
credibility rests on quietly becomes a training set for model selection.

## Decision

A separate pipeline, `scripts/compare.py` -> `artifacts/comparison_<source>.json`,
runs every (feature set x boosting config) pair through the same temporal holdout
`train.py` uses, under three rules:

**Selection never sees the test fold.** Each variant is fit on an inner train
slice and scored on an inner validation slice, both carved out of the train
window with the same leakage-safe split function used everywhere else, so
validation is later in time than what the model saw. The winner is chosen on
validation PR-AUC alone. The test fold is scored for every variant and printed,
and it chooses nothing.

**Every variant is reported, winners and losers.** The artifact holds all eight
rows. It also records `best_on_test_not_selected` - the variant with the best
test number that validation did not pick - on purpose, priced with its own
interval, so the one row a reader is most tempted to quote is sitting there with
the reason it is not the headline attached.

**"Better" is a paired bootstrap, not a difference of two rounded numbers.** The
gap between a challenger and the incumbent is bootstrapped with both models scored
on the same resampled rows, which cancels the fold-draw variance they share.
Unpaired intervals on two PR-AUCs from one 75-positive fold overlap almost always
and would read as "no difference" even when one model wins every draw; pairing
leaves the part that is actually about the models.

## Consequences

- **The velocity result on ULB is negative, and it is kept.** Validation chose the
  incumbent. Velocity on the default config hurt: paired delta -0.0119 PR-AUC, 95%
  CI [-0.0251, -0.0001], winning 2% of resamples. The reason is structural, not a
  bug: ULB publishes no card id, so `data.py` gives every row its own entity, the
  per-entity velocity family degenerates to all-zeros and is dropped, and the
  stream-level counts that remain are largely redundant with the V1..V28 PCA
  components. The family needs a dataset with real entities (Sparkov, IEEE-CIS) to
  show its worth. A negative result published is worth more here than a flattering
  one hidden.
- **The default config is now a measured choice.** `gbdt_shallow` posts the best
  test number (0.7614) but its paired interval covers zero, and validation ranked
  the incumbent first, so the headline config stands - now because it was checked,
  not assumed.
- **The headline is untouched.** The incumbent's bootstrap is recomputed inside
  this pipeline and lands on the committed CI [0.6214, 0.8232] exactly, which is
  the check that the comparison is describing the same model. PR-AUC 0.7278 in
  `metrics_ulb_creditcard.json` is byte-identical to before.
- The comparison shares the bootstrap's limits (ADR 007): sampling noise only, one
  fold, independent resampling. It answers "did this feature or config help on this
  fold", not "will it help on the next quarter's fraud".

## Rejected

- **Picking the best test PR-AUC out of eight.** The failure this ADR exists to
  prevent. It is the single most common way a leakage-safe repo leaks.
- **Cross-validated selection over the whole set.** A k-fold CV that shuffles rows
  across time would reintroduce exactly the temporal leakage ADR 001 removed. The
  inner split is the same time-ordered, entity-whole split as the outer one.
- **Only reporting the winner.** A comparison that publishes its winner and drops
  its losers is an advertisement. The losing rows are the evidence that the winner
  was chosen honestly.
- **Vendoring xgboost/lightgbm for the config sweep.** Same reasoning as
  registry.py's existing note: a dependency the repo neither installs nor tests
  would be dead code. The two extra configs are the same in-tree
  `HistGradientBoostingClassifier` with regularization dialed down and up.
