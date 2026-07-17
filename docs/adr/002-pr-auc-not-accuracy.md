# ADR 002: PR-AUC as the headline metric, never accuracy, ROC-AUC demoted

Status: accepted

## Context

The ULB test fold is 0.132% fraud. At that base rate:

- **Accuracy** is a broken instrument: the constant answer "legit" scores 99.87%
  while catching zero fraud. Every "99.9% accurate fraud model" headline on this
  dataset is reporting the base rate back as an achievement.
- **ROC-AUC** is misleading in a subtler way: it is insensitive to the false-
  positive count that actually matters. With 56,886 legit rows, a model can
  flag thousands of false positives and barely dent its ROC-AUC, because the
  false-positive RATE stays tiny. A fraud desk lives in precision space -
  "of what you flagged, how much was real?" - and ROC-AUC does not answer it.

## Decision

PR-AUC (average precision) is the headline, always reported against its own
no-skill baseline, which for PR-AUC is the positive rate of the fold (0.0013
here) - so the honest claim is "0.7278 vs 0.0013 no-skill", a ~550x lift, not a
free-floating decimal. The full coverage/precision/recall table at operating
thresholds sits beside it, because an area under a curve is not an operating
point.

Both rejected metrics are nonetheless **computed and committed as evidence**,
under `demoted_metrics` in every metrics file, with the caveat written into the
artifact beside them. An ADR that argues a metric misleads and never computes it
is asserting, not measuring, which is the exact failure this repo exists to
avoid. Demoted means "not the headline", not "not measured".

## Evidence (measured on ULB, `artifacts/metrics_ulb_creditcard.json`)

Same model, same fold, same predictions as the 0.7278 above:

| | value | no-skill baseline |
|---|---|---|
| PR-AUC (headline) | **0.7278** | 0.0013 (the fold's fraud rate) |
| ROC-AUC (demoted) | **0.9740** | 0.5, at any imbalance |
| accuracy (demoted) | **0.9971** | - |
| accuracy, always predict "legit" | **0.9987** | catches 0 of 75 frauds |

Two things fall out of that table, and they are the entire argument:

1. **ROC-AUC 0.9740 and PR-AUC 0.7278 describe identical predictions.** One says
   excellent, the other says decent. Both are correctly computed. The gap is not a
   disagreement about the model, it is a disagreement about what counts as a
   mistake: `FPR = FP / (FP + TN)` carries 56,886 easy negatives in its
   denominator, so hundreds of false positives barely move it, while precision
   (`TP / (TP + FP)`) has no `TN` term at all and feels every one of them. Note
   also the baselines: PR-AUC's moves with the fold, ROC-AUC's is 0.5 no matter how
   imbalanced the data gets, which is precisely why 0.9740 feels impressive and
   means less than it looks.
2. **This model's accuracy is worse than doing nothing.** 0.9971 against 0.9987
   for the constant answer, which catches zero fraud. That is not a defect being
   confessed; it is the strongest available proof that accuracy is a broken
   instrument at this base rate, and it is more persuasive coming from our own
   model than from any argument.

## Consequences

- Numbers look worse than the accuracy-quoting competition. Deliberate.
- The no-skill baseline changes per fold (it is the fold's fraud rate), so every
  metrics file records both.
- The demoted numbers are pinned in `scripts/verify_repro.py` like any other
  published figure: if this repo prints a number, that number reproduces.
- Publishing ROC-AUC invites "so you are behind the 0.99 notebooks". The answer is
  in the table: we are not behind them, we are reporting the metric that cannot
  flatter us, and here is theirs too.

## Rejected

- **Accuracy:** above. Reported as demoted evidence only.
- **ROC-AUC as headline:** above. Reported as demoted evidence only. Earlier
  revisions of this ADR declined to compute it at all; that was wrong. Refusing to
  measure a metric is not the same as refusing to be led by it, and the unmeasured
  version left this repo's central claim resting on an argument instead of a
  number.
- **F1 at a fixed 0.5 cutoff:** collapses the entire operating range into one
  arbitrary threshold; the reject-knob table is that range, made explicit.
