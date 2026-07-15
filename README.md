# LedgerSentry

Real-time financial-transaction fraud detection with a tunable reject-to-review option.
A gradient-boosted classifier that scores each transaction and, when it isn't confident
either way, abstains and returns `"review"` instead of guessing. It is the FlowSentry
architecture (my real-time network-intrusion-detection flagship) retargeted from network
flows to financial transactions: same reject-option idea, same leakage-safe evaluation
discipline, same honesty rules, different domain. Build once, two flagships.

> **Data note, read this first.** Every number below is measured on a **deterministic
> synthetic transaction fixture** (seeded, 8,000 rows, ~1% fraud), not real financial
> data. It exists so this pipeline, its tests, and CI are green with zero network access
> and zero Kaggle-gated downloads. The real-data run (Sparkov / IEEE-CIS / ULB / Amazon
> `fraud-dataset-benchmark`) is a documented next step - see "Data" below and
> `HANDOFF.md`. Every metrics artifact this repo produces is tagged `is_synthetic` for
> exactly this reason, and nothing here is trained on, or points at, real payments.

## Money boundary

This is fraud-**detection** software and research only. No trading, no moving money, no
personalized financial advice, anywhere in this repo.

## Why a reject-to-review option

Most fraud-model demos report one accuracy number on a dataset that's 99% "legit" - a
model that predicts "legit" every single time scores 99% accuracy and catches nothing.
The honest metric on data like this is **PR-AUC** (area under the precision-recall
curve), not accuracy, and the honest engineering answer to uncertainty is the same one
FlowSentry uses for network flows: don't force a guess on a transaction the model isn't
sure about. Score it, and if the confidence is below a chosen bar, route it to `review`
(a human, or a heavier downstream check) instead. Sweep that bar and you get a
coverage-vs-precision curve instead of a single number - that curve, not any one
accuracy figure, is the product.

## Architecture

```
   data/*.csv (real, optional)          make_synthetic() (deterministic fallback)
   Sparkov / IEEE-CIS / ULB / FDB                 seeded, ~1% fraud, has a
              |                                    timestamp + entity_id column
              +------------------+------------------+
                                 v
                    data.py -- canonical schema
          (transaction_id, timestamp, entity_id, amount,
                     category, is_fraud, f_*)
                                 |
                                 v
                 temporal_grouped_split (leakage-safe:
              every entity_id lands ENTIRELY in train or
                test, entities ordered by first-seen time)
                        /                    \
                       v                      v
                 train split             test split
                       |                      |
                       v                      |
      build_preprocessor().fit_transform      |
        (one-hot category + numeric           |
         passthrough, TRAIN ONLY)             v
                       |            preprocessor.transform (test)
                       v                      |
      FraudDetector.fit()                     |
    (HistGradientBoostingClassifier,          |
     balanced sample weights, TRAIN ONLY)     |
                       |                      |
                       +----------+-----------+
                                  v
                    .predict_proba_fraud() on held-out test
                                  |
                                  v
                confidence = max(p_fraud, 1 - p_fraud)
                                  |
                    threshold sweep (the reject knob)
                          /                \
                         v                  v
              coverage x precision      "fraud" / "legit"
                   table                 / "review" (abstain)
```

Built today (F1, this milestone): the canonical loader, the deterministic synthetic
fixture, the leakage-safe grouped/temporal split, the baseline model with the
reject-to-review knob, and the metrics/model-card pipeline. A live `/predict` endpoint,
a streaming replay, and a dashboard with the knob as a slider are F2 (roadmap below) -
this milestone is offline training + evaluation only, exactly like FlowSentry's own
Week-1 milestone was before its FastAPI service landed.

## Results (real, measured, synthetic fixture)

Dataset: deterministic synthetic transactions, 8,000 rows, seeded (`seed=42`).
Leakage-safe split: 6,419 train rows / 1,581 test rows, grouped by account
(`entity_id`) with zero accounts shared between train and test, entities ordered by
first-seen time so the split is also approximately temporal. Test fraud rate 0.95%
(15 fraud rows of 1,581).

**PR-AUC on the imbalanced holdout: 0.7884** (random/no-skill baseline on this fold:
0.0095 - the model's PR-AUC is roughly 83x the no-skill baseline). Not claimed to be
comparable to the ~0.86-0.88 published XGBoost numbers on real card data (different,
synthetic dataset) - see `docs/model_card.md` for why a number well short of 1.0 is the
expected, honest result here, not a shortfall.

**Coverage vs precision (the reject-to-review knob working):**

| Review threshold | Coverage | Sent to review | Flagged fraud | Precision on flagged |
|---|---|---|---|---|
| 0.50 | 100.00% | 0 | 13 | 76.92% |
| 0.70 | 99.87% | 2 | 12 | 83.33% |
| 0.90 | 99.81% | 3 | 11 | 81.82% |
| 0.95 | 99.68% | 5 | 10 | 90.00% |
| 0.99 | 99.30% | 11 | 9 | 88.89% |

Full 8-row table (including 0.60, 0.80, 1.00) in `docs/model_card.md`, generated fresh
by every run of `scripts/train.py` from `artifacts/metrics.json`. Reading it: at the
loosest setting the model auto-decides every transaction, right on 77% of what it flags
as fraud; tightening the knob sends a handful of the most uncertain transactions to
review (11 of 1,581 at the strictest setting shown) and lifts flagged-fraud precision
into the high 80s/low 90s.

## Quickstart

```bash
git clone <this repo, once published>
cd ledgersentry
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                 # optional - scripts/train.py works without it

python scripts/train.py          # trains on the synthetic fixture (no download needed)
                                  # writes artifacts/ledgersentry.joblib + metrics.json
pytest                           # run the test suite (~15 tests, ~10-20s)
ruff check .                     # lint
```

## Data

`src/ledgersentry/data.py` is dataset-agnostic: drop a real file in `data/` and
`load()` uses it automatically instead of the synthetic fallback. None of these are
downloaded by this repo - IEEE-CIS and the Amazon benchmark are Kaggle/auth-gated, and
even the open ones are tens to hundreds of MB, so fetching is a manual step.

| Source | Why | Drop it at | Get it from |
|---|---|---|---|
| **Sparkov** (kartik2112, Kaggle) | streaming/real-time demo - has genuine timestamps, 1.85M simulated transactions, 1,000 customers x 800 merchants | `data/fraudTrain.csv` [+ `fraudTest.csv`] | [kaggle.com/datasets/kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) |
| **IEEE-CIS Fraud Detection** (Vesta) | headline benchmark - 590,540 real e-commerce transactions, 393 features | `data/train_transaction.csv` [+ `train_identity.csv`] | [kaggle.com/c/ieee-fraud-detection](https://www.kaggle.com/c/ieee-fraud-detection) |
| **Amazon `fraud-dataset-benchmark`** | comparability - a standardized multi-dataset benchmark harness | `data/fdb_train.csv` [+ `fdb_test.csv`] (export `obj.train.to_csv(...)` yourself; the package has no file-export of its own) | [github.com/amazon-science/fraud-dataset-benchmark](https://github.com/amazon-science/fraud-dataset-benchmark) |
| **ULB Credit Card Fraud** | classic leakage-safe baseline - 284,807 European transactions, 492 fraud | `data/creditcard.csv` | [kaggle.com/mlg-ulb/creditcardfraud](https://www.kaggle.com/mlg-ulb/creditcardfraud) |

Grounding for why these four and why the market/model case holds up:
`../Transcendent/projects/portfolio-site/FINTECH_PLAN.md` (private planning doc, not
part of this repo).

## Repository layout

```
src/ledgersentry/
  data.py     canonical schema, real-source loaders, synthetic fixture, leakage-safe split
  model.py    FraudDetector (HistGradientBoostingClassifier + reject-to-review knob)
  train.py    end-to-end pipeline: load -> split -> fit -> evaluate -> write artifacts
scripts/
  train.py    CLI entry point: python scripts/train.py
tests/        synthetic-fixture tests (determinism, split leakage, PR-AUC range, reject knob)
docs/
  model_card.md   full measured results + honest limitations
artifacts/     ledgersentry.joblib (gitignored) + metrics.json (committed - the source
               of truth every number above is copied from)
```

## Roadmap

**F1: offline baseline (this milestone, DONE)**
- [x] Dataset-agnostic loader: real sources when present, deterministic synthetic
      fallback otherwise
- [x] Leakage-safe grouped + approximately-temporal split, `entity_id` excluded from
      features
- [x] Baseline `FraudDetector` (HistGradientBoostingClassifier, balanced sample
      weights), reject-to-review knob, coverage-vs-precision curve
- [x] PR-AUC (not accuracy) reported against its own random-baseline for context
- [x] Tests (determinism, no group leakage, PR-AUC range, reject-knob behavior) + CI
      (ruff + pytest + a synthetic training smoke test)

**F2: streaming demo + dashboard (not built)**
- [ ] `/predict` FastAPI endpoint: score a transaction, review threshold as a request
      parameter, mirroring FlowSentry's `/predict` + `/curve`
- [ ] Replay Sparkov transactions in timestamp order -> scorer -> live feed, with
      measured latency (mirrors FlowSentry's Week-2 real-time pipeline)
- [ ] Dashboard with the review-threshold knob as a live slider (coverage vs precision,
      live), mirroring FlowSentry's Streamlit reject-knob demo
- [ ] Dockerfile + docker-compose, matching FlowSentry's container setup

**F3+: real-data run + hardening (not built)**
- [ ] Train and report on Sparkov / IEEE-CIS / ULB (real PR-AUC, replacing/joining the
      synthetic-fixture numbers, never silently swapped in for them)
- [ ] Drift monitoring, load test with real latency/throughput numbers, threat-model
      note - mirroring FlowSentry's Week-3 hardening pass

## Attribution

- The reject-to-review knob is the same architecture as FlowSentry's two-stage reject
  option, from my accepted SECRYPT 2026 paper on hierarchical UDP/QUIC intrusion
  detection, retargeted here from network flows to financial transactions.
- Sparkov, IEEE-CIS, the Amazon `fraud-dataset-benchmark`, and the ULB credit-card set
  are public/synthetic datasets from their respective authors - see the Data table
  above for sources. None are bundled in this repo.
- Built with scikit-learn, pandas, and joblib.

## License

MIT, see [LICENSE](LICENSE).
