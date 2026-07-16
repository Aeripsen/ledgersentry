# ADR 006: calibration as a separate pipeline, Platt by structural rule

Status: accepted

## Context

The raw model ranks well and lies about probabilities: test-fold Brier 0.002188,
worse than predicting the 0.13% base rate for every row (0.001315), ~8x
overconfident in its top reliability bin. Uncalibrated units are why the review
knob cliffs at 0.99. But a calibrator fit on the model's own training data just
certifies the overconfidence, and carving honest calibration data out of the
train window changes the model - and therefore the committed headline numbers.

## Decision

Calibration is its own pipeline (`scripts/calibrate.py`), never part of
`train.py`: fit on the first 80% of the train window, calibrate on the
temporally-later 20%, evaluate on the same untouched test fold as every other
number. The headline model keeps its full train window and its committed
metrics.

The SHIPPED calibrator must be strictly monotone - structurally incapable of
reordering scores. That admits Platt scaling and excludes isotonic regression,
and the rule is structural rather than an empirical bake-off for a measured
reason: isotonic evaluated on its own calibration slice is optimistic by
construction, so no cal-slice comparison can catch its failure. The ULB run
demonstrates it exactly - isotonic self-scored 0.7585 cal-slice PR-AUC, then
dropped the test fold's PR-AUC from 0.7544 to 0.6728 by collapsing distinct
scores into ties (52 calibration frauds cannot pin a step function). Platt cut
Brier 0.002188 -> 0.000516 with test PR-AUC bit-identical to raw. Both
calibrators' full numbers are committed either way
(`artifacts/calibration_ulb_creditcard.json`).

## Consequences

- Two models exist conceptually: the headline (full train window, uncalibrated,
  what /predict serves) and the calibrated variant (smaller window, honest
  probabilities). The model card states which numbers belong to which.
- Calibrated probabilities make the expected-cost function meaningful: a fraud
  desk can pick the review threshold by cost instead of reading a per-dataset
  curve. Costs remain caller-supplied, always.

## Rejected

- **CalibratedClassifierCV inside train.py:** changes the headline model and
  hides the training-data cost of calibration inside a wrapper.
- **Isotonic (as the shipped calibrator):** above - measured ranking damage,
  undetectable from the calibration slice alone. Still computed and reported
  every run, as evidence rather than product.
- **Choosing the calibrator on test-fold Brier:** selection leakage; the test
  fold exists to be reported, not consulted.
