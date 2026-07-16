# ADR 003: a reject-to-review option instead of a forced binary decision

Status: accepted

## Context

A binary fraud classifier is forced to guess on exactly the transactions it
knows least about, and both wrong guesses are expensive in different currencies:
a missed fraud is a chargeback, a false flag is a blocked customer. Real fraud
operations do not run binary - they run a third outcome, human review, and their
actual daily decision is how much review capacity to spend for how much
automated precision.

## Decision

The model abstains: `confidence = max(p_fraud, 1 - p_fraud)`, and below a chosen
review threshold the decision is `review` instead of fraud/legit. The product is
the measured coverage/precision/recall table over that threshold - on the real
ULB fold, review 7.1% of traffic and automated flag precision rises 29% -> 90%
while total misses drop 12 -> 6, because uncertain frauds land in the queue
instead of being auto-cleared. Every row of that table partitions all test
frauds into caught/queued/missed so recall is never hidden behind precision.
The reject option is the same architecture as the author's SECRYPT 2026
intrusion-detection paper, retargeted from network flows to transactions.

## Consequences

- Reports are three-outcome everywhere; anything that summarizes this system
  with one number is wrong by construction.
- The knob's units were raw-model confidence, which is NOT a probability - that
  gap is measured and fixed in ADR 006 (calibration).
- Review capacity is a real cost, so the expected-cost function
  (`model.expected_cost_curve`) prices the queue explicitly, with caller-supplied
  costs only.

## Rejected

- **Single fixed threshold:** throws away the operating range a fraud desk
  actually tunes.
- **Cost-sensitive training (costs baked into the loss):** requires cost
  assumptions this repo refuses to invent; costs enter at decision time, where
  the caller owns them.
