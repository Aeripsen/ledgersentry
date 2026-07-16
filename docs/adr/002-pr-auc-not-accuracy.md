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

## Consequences

- Numbers look worse than the accuracy-quoting competition. Deliberate.
- The no-skill baseline changes per fold (it is the fold's fraud rate), so every
  metrics file records both.

## Rejected

- **Accuracy:** above.
- **ROC-AUC as headline:** above; still computed nowhere rather than reported
  and caveated, to keep one honest headline.
- **F1 at a fixed 0.5 cutoff:** collapses the entire operating range into one
  arbitrary threshold; the reject-knob table is that range, made explicit.
