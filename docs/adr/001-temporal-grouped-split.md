# ADR 001: strict temporal + entity-grouped evaluation split

Status: accepted (in force since the first commit)

## Context

Most published fraud results on public datasets use a random split, and random
splits leak in two specific ways on transaction data:

1. **Entity leakage.** One card's transactions are heavily correlated. Shuffle
   them across train and test and the model grades itself on customers it has
   effectively memorized - it learns "card 4417 gets defrauded," not what fraud
   looks like. Production never sees this luxury: every scored transaction
   belongs to the future, mostly from entities behaving in new ways.
2. **Temporal leakage.** A random split trains on Tuesday's fraud patterns and
   tests on Monday's. Fraud is adversarial and non-stationary; knowing the
   future distribution of attacks is exactly the information a deployed model
   will not have. The related classic is resampling before splitting: SMOTE fit
   on the full set manufactures test-fold neighbors of training points and
   produces the fake "AUC 1.000" results the literature keeps having to retract.

## Decision

`temporal_grouped_split`: every `entity_id` lands entirely in train or entirely
in test, and entities are assigned in ascending first-seen order, so every train
entity is first seen no later than every test entity. All resampling/weighting
is fit strictly after the split, on train rows only (this repo uses balanced
sample weights, which never touch X at all). On ULB, which publishes no entity
key, the split degrades to purely temporal and the docs say so rather than
pretending the grouped guarantee held.

## Consequences

- The headline PR-AUC (0.7278) is lower than shuffled-split numbers on
  comparable data (~0.86-0.88 published). That is the point: it is the number
  a deployment would actually see, and the README refuses the comparison
  rather than winning it dishonestly.
- The synthetic fixture's split behavior is CI-tested (no entity overlap,
  train cohorts precede test cohorts), so the guarantee is enforced, not
  aspirational.

## Rejected

- **Random / stratified shuffle split:** leaks as above; produces better-looking,
  worse-meaning numbers.
- **Random k-fold CV for the headline:** k-fold with time-ordered data either
  shuffles (leaks) or becomes a rolling-origin scheme; rolling-origin evaluation
  is a legitimate future addition but is not what the committed headline uses.
