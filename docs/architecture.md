# Architecture

One package, two pipelines (train and serve), three deliberate seams. Decision
records live in `docs/adr/`; the cuts - what was considered and rejected as
speculative - are in [ADR 004](adr/004-serving-shape.md) and matter as much as
what shipped.

## Training pipeline

```mermaid
flowchart TD
    subgraph sources["data/ (never committed)"]
        SP["Sparkov csv"]
        IE["IEEE-CIS csv"]
        FDB["Amazon FDB csv"]
        ULB["ULB creditcard.csv"]
    end
    SYN["make_synthetic()\nseeded fixture, ~1% fraud"]
    LS["LOADERS spec table\n(data.py - first match wins,\none spec row per source)"]
    SP & IE & FDB & ULB --> LS
    LS -->|none present| SYN
    LS --> CANON["canonical frame\ntransaction_id, timestamp, entity_id,\namount, category, is_fraud, f_*"]
    SYN --> CANON
    CANON --> SPLIT["temporal_grouped_split\nentities whole, time-ordered\n(ADR 001)"]
    SPLIT --> TR["train split"] & TE["test split"]
    TR --> PRE["ColumnTransformer\nfit on TRAIN only"]
    TR --> REF["drift reference\nper-feature quantile bins"]
    PRE --> FIT["FraudDetector.fit\nmodel from registry.py,\nbalanced sample weights"]
    FIT --> EVAL["PR-AUC vs no-skill (ADR 002)\n+ reject-knob table (ADR 003)"]
    TE --> EVAL
    EVAL --> MET["artifacts/metrics_&lt;source&gt;.json"]
    FIT & PRE & REF --> ART["artifacts/ledgersentry.joblib"]
```

## Serving pipeline

```mermaid
flowchart TD
    ART["ledgersentry.joblib\n{preprocessor, model, drift_reference}"]
    ART --> CS["CompiledScorer (ADR 005)\npreprocessor compiled to numpy once;\nPandasScorer kept as parity oracle"]
    CS --> P["/predict - one row,\np99 1.73 ms measured"]
    CS --> PB["/predict/batch - vectorized,\ncapped at max_batch"]
    ART --> DR["/drift - PSI per feature\nvs frozen train reference"]
    RH["/health (liveness)\n/ready (traffic gate)"]
    CS --> STREAM["stream.py replay\n(measured per-row latency)"]
    CS --> DASH["Streamlit dashboard\n(live review-threshold slider)"]
    CAL["calibration artifact (ADR 006)\nPlatt map + reliability + Brier"]
    ART -.->|separate pipeline,\nscripts/calibrate.py| CAL
```

## The three seams (each has >= 2 real implementations today)

| Seam | Contract | Implementations in this repo |
|---|---|---|
| Data source | `LoaderSpec` + `LoaderFn` protocol (data.py) | Sparkov, IEEE-CIS, Amazon FDB, ULB (+ synthetic fallback) |
| Classifier | registry factory (registry.py) | `hist_gbdt`, `logreg` (a test registers a third at runtime to prove the seam) |
| Scoring path | `TransactionScorer` protocol (scoring.py) | `CompiledScorer` (serving), `PandasScorer` (reference + benchmark baseline) |

That "two real implementations" bar is the repo's rule for abstractions: a seam
with one implementer is speculation, and several candidates failed the bar and
were cut by name (feature-pipeline protocol, sink/alert abstraction, plugin
loading - ADR 004).

## Module map

```
src/ledgersentry/
  config.py      pydantic-settings: env > yaml > defaults; defaults = the committed run
  data.py        loaders (spec table), canonical schema, synthetic fixture, the split
  registry.py    model registry (one register() call to add a classifier)
  model.py       FraudDetector + reject knob + curve/expected-cost math
  scoring.py     TransactionScorer protocol, compiled + reference scorers
  train.py       load -> split -> fit -> evaluate -> artifacts (+ drift reference)
  calibration.py Platt/isotonic calibration pipeline (separate from train, ADR 006)
  drift.py       PSI vs frozen training reference
  bench.py       committed latency/throughput harness (writes benchmark.json)
  service.py     FastAPI: predict/batch/drift/health/ready/curve, JSON logs
  stream.py      timestamp-ordered replay of the held-out test split
```
