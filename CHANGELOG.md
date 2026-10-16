# Changelog

Grouped by the dates in `git log`; the commit messages hold the detail. No release has been
tagged; `pyproject.toml` says 0.1.0.

## Unreleased (2026-10-16)

- `scripts/headline.py` (`make headline`): recomputes PR-AUC, the reject-knob table and the
  business-case policies from the committed per-transaction scores in seconds, offline, and
  exits 1 if any differs from the committed artifacts. Tested in `tests/test_headline.py`,
  including that a hand-edited metric fails it.
- README cut to a three-minute read: the problem, the measured numbers with the command that
  reproduces them, why a temporal holdout, the review knob explained, architecture,
  quickstart, limitations. Every other section moved verbatim to `docs/DETAILS.md`.
- Docstrings on the request models, `/predict`, `/curve`, the scorer contract,
  `FraudDetector.fit` and `Settings`. `CITATION.cff` and this changelog added.

## 2026-09-30: operations view, live demo, deployment and load testing

- Business case: the knob in alerts, review load and fraud caught per 10,000 transactions,
  with each effect credited to the policy change that produces it and paired bootstrap CIs.
- Live demo on GitHub Pages over the real held-out fold; CI regenerates the scores from the
  public data, drives the page in headless Chromium, and Pages deploys only after all of ci.
- Deployment: Kubernetes on kind in CI with a rolling restart under load, a preStop
  connection drain (A/B measured), Terraform applied to kind, docker compose smoke test.
- MLflow tracking for training and both comparisons, and an Evidently drift and performance
  report over two time windows of the real split.
- HTTP load test under concurrent clients with provenance in every result; found OpenMP
  oversubscription on the request path and pinned one thread per process.

## 2026-07-20 to 2026-07-23: comparisons, attribution, deploy manifests

- Rolling velocity features and two more boosting configs, compared on the real fold with
  selection on an inner validation slice: velocity did not help, and that result is kept.
- Expected cost curve committed as an artifact instead of asserted figures.
- LightGBM as an optional model: it beats the shipped model on this fold (paired interval
  excludes zero); the headline deliberately stays on the shipped model.
- Exact SHAP attribution for the shipped artifact.
- Self-contained Docker image plus Render, Kubernetes and Terraform manifests; CI badge.

## 2026-07-14 to 2026-07-16: baseline, first real-data run, hardening

- Offline baseline: dataset-agnostic loaders, leakage-safe temporal split, reject-to-review
  model; FastAPI service, timed stream replay and Streamlit dashboard.
- First real-data run on ULB: PR-AUC 0.7278 on the temporal holdout, recall reported beside
  precision, CI coverage for all four loaders.
- Reproducibility: exact dependency pins and `make reproduce` with a byte-identity check.
- Compiled scoring path, benchmarked, with a CI regression guard.
- Model registry, pydantic-settings config, Platt calibration, PSI drift, `/ready`,
  structured logging, threat model, ADRs and a mermaid architecture doc.
- ROC-AUC and accuracy measured and published as demoted metrics; bootstrap 95% CIs on the
  headline; the reject knob's two cuts decoupled.
