# LedgerSentry

Real-time financial-transaction fraud detection with a tunable reject-to-review option.
A gradient-boosted classifier that scores each transaction and, when it isn't confident
either way, abstains and returns `"review"` instead of guessing. It is the FlowSentry
architecture (my real-time network-intrusion-detection flagship) retargeted from network
flows to financial transactions: same reject-option idea, same leakage-safe evaluation
discipline, same honesty rules, different domain.

**The 90-second version:**

- **Real result:** PR-AUC **0.7278** on the ULB credit-card fraud set (284,807 real
  anonymized transactions, 492 fraud, 0.17%) with a strictly temporal holdout - train
  on the first ~40 hours, test on the last ~7.6 hours, nothing from the future leaks
  back. No-skill baseline on that fold is 0.0013, so that is roughly a 550x lift.
- **The knob is the product:** fully automated, 29% of flagged transactions are truly
  fraud; send the most uncertain 7.1% of traffic to human review and automated flags
  become **89.8% precise**. The full measured coverage-vs-precision table is below.
- **The edge is honest evaluation:** leakage-safe grouped + temporal splits, PR-AUC
  instead of accuracy on 99.8%-legit data, and a reject option instead of forced
  guesses - the same core as my SECRYPT 2026 paper on intrusion detection with a
  reject option. A defensible measured 0.73 beats a fake 0.99, and hiring managers
  and reviewers know the difference.
- **Everything is reproducible:** `pip install -r requirements.txt`, drop
  `data/creditcard.csv` in (one public URL, no login - see Data), run
  `python scripts/train.py`. Without the file, the same command runs a deterministic
  synthetic fixture so tests and CI stay green offline. Real and synthetic numbers
  are never mixed: every metrics file is tagged `is_synthetic`.

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

Built in F1: the canonical loader, the deterministic synthetic fixture, the
leakage-safe grouped/temporal split, the baseline model with the reject-to-review
knob, and the metrics/model-card pipeline. Built in F2: a live
`/predict` + `/health` + `/curve` FastAPI service, a one-row-at-a-time streaming
replay with real measured latency, a Streamlit dashboard with the review-threshold
knob as a live slider, and a Dockerfile/docker-compose serving both - mirroring
FlowSentry's own Week-2 milestone.

## Results - REAL DATA (ULB credit-card fraud, measured 2026-07-15)

Dataset: ULB "Credit Card Fraud Detection" - 284,807 real anonymized European card
transactions over 2 days, 492 fraud (0.173%). ULB publishes no card/customer id, so
the grouped split degrades (documented in `data.py`) to a **pure temporal split**:
227,846 train rows (417 fraud) / 56,961 test rows (75 fraud, 0.132%) - the model is
evaluated only on the final ~7.6 hours of transactions it has never seen.

**PR-AUC on the real imbalanced holdout: 0.7278** (no-skill baseline on this fold:
0.0013, roughly a 550x lift). Published XGBoost-class numbers on real card data run
~0.86-0.88, but on random (non-temporal) splits of different datasets - not directly
comparable, and not claimed to be. See `docs/model_card.md` for the full honesty
notes. Source of truth: `artifacts/metrics_ulb_creditcard.json`
(`is_synthetic: false`).

**Coverage vs precision on real data (the reject-to-review knob working):**

| Review threshold | Coverage | Sent to review | Flagged fraud | Precision on flagged |
|---|---|---|---|---|
| 0.50 | 100.00% | 0 | 216 | 29.17% |
| 0.70 | 99.45% | 312 | 129 | 44.19% |
| 0.80 | 98.98% | 583 | 105 | 53.33% |
| 0.90 | 97.22% | 1,582 | 73 | 73.97% |
| 0.95 | 92.91% | 4,039 | 59 | 89.83% |

Reading it: fully automated, 29% of flagged transactions are truly fraud; route the
most uncertain 7.1% of traffic to review and the automated flags become 89.8%
precise. That 3x precision lift for a bounded human-review budget is the product.
(Above 0.95 the uncalibrated confidence cliff sends almost everything to review -
measured, shown in the model card's full 8-row table, and called out as a limitation
rather than hidden.)

### Synthetic fixture (offline CI baseline, labeled synthetic)

With no real file in `data/`, the same pipeline runs a deterministic seeded fixture
(8,000 rows, ~1% fraud) so tests and CI are green with zero downloads: PR-AUC
**0.7884** against a 0.0095 no-skill baseline (`artifacts/metrics_synthetic.json`,
`is_synthetic: true`). Full table and caveats in `docs/model_card.md` - synthetic
numbers are a pipeline proof, not a benchmark claim, and are never mixed with the
real ones.

## Live serving - measured latency (F2)

> Same data note as above: this is the **synthetic fixture's** held-out test split
> (1,581 rows) replayed one row at a time through the trained artifact - not real
> Sparkov data (still not downloaded, still Kaggle-gated). It plays the same
> narrative role (real timestamps, chronologically sorted) but these are not
> real-transaction-volume numbers. Real output, pasted verbatim, from:
> `python scripts/stream.py --n 0` on this machine, one run, single-thread, no GPU.

```
[stream] replaying 1581 transactions from the 'synthetic' test split (review_threshold=0.0)
...
[rows   ] 1581   fraud=13  legit=1568  review=0
[latency] per-row preprocess+decide  mean=12.072 ms  p50=11.074 ms  p95=12.986 ms  p99=13.946 ms
[through] 76 rows/sec over 20.67s wall (single-thread, this machine, data_source=synthetic)
```

Per-row latency is dominated by pandas/`ColumnTransformer` overhead on a 1-row frame,
not the gradient-boosted model itself - `HistGradientBoostingClassifier.predict_proba`
on one row is fast; building and one-hot-encoding a fresh 1-row DataFrame each time is
the actual cost. A production version would batch or reuse a warm encoder; this replay
deliberately does neither, to measure the honest one-row-at-a-time worst case the way
FlowSentry's `stream.py` does for network flows.

## Quickstart

```bash
git clone <this repo, once published>
cd ledgersentry
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                 # optional - scripts/train.py works without it

python scripts/train.py          # no data file present -> synthetic fixture (offline)
                                  # writes artifacts/ledgersentry.joblib + metrics.json
pytest                           # run the test suite (27 tests, ~20s)
ruff check .                     # lint

# reproduce the REAL ULB numbers (one public file, no login):
curl -o data/creditcard.csv https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
python scripts/train.py          # detects the file, trains + evaluates on real data,
                                  # writes artifacts/metrics_ulb_creditcard.json
```

## Serving + dashboard

Requires `artifacts/ledgersentry.joblib` to exist first - run `python scripts/train.py`
(above) at least once.

```bash
# FastAPI service: /health, /predict, /curve
uvicorn ledgersentry.service:app --app-dir src --reload
# in another shell:
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"amount": 420.85, "category": "travel", "timestamp": "2026-03-04T02:11:00"}, "review_threshold": 0.9}'

# Streaming replay: real per-row latency, printed at the end (see "Live serving" above)
python scripts/stream.py --n 0

# Streamlit dashboard: review-threshold slider, live coverage/precision, live feed
streamlit run dashboard/app.py

# Or both services in one container stack (api:8000, dashboard:8501):
# (honesty note: the image has not been build-tested yet - Docker was unavailable
#  on the build machine; the files mirror FlowSentry's known-good setup)
docker compose up --build
```

`/predict` derives its expected input columns from the FITTED preprocessor inside the
loaded artifact (not a hardcoded schema), so it isn't locked to the synthetic fixture's
columns - swap in a real dataset (see "Data" below), retrain, and the same endpoint
serves whatever `f_*` features that source produced. Missing numeric fields default to
0, missing `category` defaults to `"unknown"`.

## Data

`src/ledgersentry/data.py` is dataset-agnostic: drop a real file in `data/` and
`load()` uses it automatically instead of the synthetic fallback. No dataset is
committed to or downloaded by this repo - fetching is a manual step. All four loaders
are exercised in CI against tiny true-schema fixture CSVs (`tests/test_loaders.py`),
so a loader regression turns CI red without shipping any real data.

| Source | Why | Drop it at | Get it from |
|---|---|---|---|
| **ULB Credit Card Fraud** | classic baseline, the real-data run above - 284,807 European transactions, 492 fraud | `data/creditcard.csv` | no login needed: `curl -o data/creditcard.csv https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv` (TensorFlow's hosted copy of the [Kaggle set](https://www.kaggle.com/mlg-ulb/creditcardfraud); verify 284,807 rows / 492 fraud after download) |
| **Sparkov** (kartik2112, Kaggle) | streaming/real-time demo - has genuine timestamps, 1.85M simulated transactions, 1,000 customers x 800 merchants | `data/fraudTrain.csv` [+ `fraudTest.csv`] | [kaggle.com/datasets/kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (Kaggle login) |
| **IEEE-CIS Fraud Detection** (Vesta) | headline benchmark - 590,540 real e-commerce transactions, 393 features | `data/train_transaction.csv` [+ `train_identity.csv`] | [kaggle.com/c/ieee-fraud-detection](https://www.kaggle.com/c/ieee-fraud-detection) (Kaggle login) |
| **Amazon `fraud-dataset-benchmark`** | comparability - a standardized multi-dataset benchmark harness | `data/fdb_train.csv` [+ `fdb_test.csv`] (export `obj.train.to_csv(...)` yourself; the package has no file-export of its own) | [github.com/amazon-science/fraud-dataset-benchmark](https://github.com/amazon-science/fraud-dataset-benchmark) |

## Repository layout

```
src/ledgersentry/
  data.py     canonical schema, real-source loaders, synthetic fixture, leakage-safe split
  model.py    FraudDetector (HistGradientBoostingClassifier + reject-to-review knob)
  train.py    end-to-end pipeline: load -> split -> fit -> evaluate -> write artifacts
  service.py  FastAPI app: /health, /predict, /curve
  stream.py   one-row-at-a-time replay of the held-out test split, real measured latency
dashboard/
  app.py      Streamlit dashboard: review-threshold slider, live coverage/precision,
              live inference metrics, alert/review feed
scripts/
  train.py    CLI entry point: python scripts/train.py
  stream.py   CLI entry point: python scripts/stream.py
tests/        synthetic-fixture, loader, and service tests (determinism, split
              leakage, PR-AUC range, reject knob, /health + /predict + /curve, and
              all four real-dataset loaders against tiny true-schema fixture CSVs
              in tests/fixtures/)
docs/
  model_card.md   full measured results (real + synthetic, clearly separated) +
                  honest limitations
artifacts/     ledgersentry.joblib (gitignored) + committed metrics: metrics.json
               (latest run), metrics_ulb_creditcard.json (the real ULB run),
               metrics_synthetic.json (the offline CI fixture)
Dockerfile / docker-compose.yml   one image, two services (api:8000, dashboard:8501)
```

## Roadmap

**F1: offline baseline (DONE)**
- [x] Dataset-agnostic loader: real sources when present, deterministic synthetic
      fallback otherwise
- [x] Leakage-safe grouped + approximately-temporal split, `entity_id` excluded from
      features
- [x] Baseline `FraudDetector` (HistGradientBoostingClassifier, balanced sample
      weights), reject-to-review knob, coverage-vs-precision curve
- [x] PR-AUC (not accuracy) reported against its own random-baseline for context
- [x] Tests (determinism, no group leakage, PR-AUC range, reject-knob behavior) + CI
      (ruff + pytest + a synthetic training smoke test)

**F2: streaming demo + dashboard (DONE)**
- [x] `/predict` FastAPI endpoint: score a transaction, review threshold as a request
      parameter, mirroring FlowSentry's `/predict` + `/curve`; expected columns read
      off the fitted preprocessor, not hardcoded to the synthetic schema
- [x] Replay the held-out test split in timestamp order -> scorer -> live feed, with
      measured latency (real Sparkov still not downloaded - Kaggle-gated - so this
      replays the synthetic fixture's test split, honestly labeled; see "Live serving"
      above)
- [x] Dashboard with the review-threshold knob as a live slider (coverage vs precision,
      live), mirroring FlowSentry's Streamlit reject-knob demo
- [x] Dockerfile + docker-compose, matching FlowSentry's container setup (not yet
      build-tested - Docker was unavailable on the build machine; flagged below)

**F3: real-data run (ULB DONE, this milestone)**
- [x] Train and report on real data: ULB credit-card fraud, 284,807 transactions,
      temporal holdout, PR-AUC 0.7278 - reported beside the synthetic-fixture
      numbers, never silently swapped in for them
- [x] CI tests for all four real-dataset loaders (tiny true-schema fixtures in
      `tests/fixtures/`), including a regression test for the FDB amount-column
      mapping
- [ ] Sparkov full run (the streaming story - Kaggle-gated download) and IEEE-CIS
      full run (the headline benchmark)

**F4+: hardening (not built)**
- [ ] Confidence calibration (train-only CalibratedClassifierCV) so review
      thresholds are portable across datasets
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
