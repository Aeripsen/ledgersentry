# Convenience targets. Every one is a thin wrapper over a plain command that is
# also documented in the README, so Windows users without make lose nothing.

PY ?= python

.PHONY: install install-analysis test lint train reproduce bench loadtest loadtest-profile bootstrap compare compare-boosters shap cost business business-verify demo-data demo-verify site-check serve dashboard k8s-e2e tf-kind install-mlops drift-report mlflow-ui k8s-schema compose-smoke

install:
	$(PY) -m pip install -r requirements.txt

# Optional extras for the booster head-to-head and the SHAP report. shap goes
# in --no-deps because its numba dependency caps numpy below this repo's pin;
# the why lives in requirements-analysis.txt.
install-analysis:
	$(PY) -m pip install -r requirements-analysis.txt
	$(PY) -m pip install --no-deps shap==0.52.0

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts dashboard

train:
	$(PY) scripts/train.py

# Regenerate the real ULB numbers and fail unless they match the committed
# metrics byte for byte. Needs data/creditcard.csv (see README "Data" - one
# public URL, no login).
reproduce:
	$(PY) scripts/train.py
	$(PY) scripts/verify_repro.py

bench:
	$(PY) scripts/bench.py

# HTTP load test, both halves of the A/B in one command: the shipped arm (the
# Dockerfile's OMP_NUM_THREADS=1) and the before arm (thread variables unset),
# alternating ABBA, 3 rounds at 1 and 4 workers on /predict, 2 rounds of the
# 100-row batch endpoint, then the py-spy profiles the README quotes.
# Writes artifacts/loadtest_ab/<endpoint>/*.json + summary.json and
# artifacts/profiles/. Needs requirements-loadtest.txt. README "Load test".
loadtest:
	$(PY) scripts/loadtest.py --ab-rounds 3 --ab-workers 1,4
	$(PY) scripts/loadtest.py --endpoint batch --ab-rounds 2 --ab-workers 1
	$(MAKE) loadtest-profile

loadtest-profile:
	$(PY) scripts/loadtest.py --levels 4 --pyspy-at 4 --label profile_predict_w1_omp1 --out artifacts/profiles/profile_predict_w1_omp1.json
	$(PY) scripts/loadtest.py --endpoint batch --levels 4 --pyspy-at 4 --label profile_batch_w1_omp1 --out artifacts/profiles/profile_batch_w1_omp1.json

# 95% confidence intervals on the headline, so the four decimals it prints get
# read with the uncertainty they actually carry.
bootstrap:
	$(PY) scripts/bootstrap.py

# Feature-set x boosting-config comparison on the same temporal holdout,
# selected on an inner validation slice. Reports every variant.
compare:
	$(PY) scripts/compare.py

# LightGBM vs the incumbent hist_gbdt on the same holdout, paired bootstrap on
# the delta. Needs `make install-analysis` first.
compare-boosters:
	$(PY) scripts/compare_boosters.py

# SHAP attribution on the shipped artifact -> shap_<source>.json + summary PNG.
# Needs `make install-analysis` first.
shap:
	$(PY) scripts/shap_report.py

# Expected cost per review threshold on calibrated scores, priced under
# several illustrative cost triples so the optimum's dependence on the
# assumptions is visible.
cost:
	$(PY) scripts/cost.py

# The knob in operations units: alerts and false alerts per 10k transactions,
# review load, frauds caught / in review / missed, and the share of fraud by
# the dataset's own Amount. Counts only, nothing priced. Fails unless its counts
# match the committed metrics file. business-verify requires a byte match.
business:
	$(PY) scripts/business_case.py

business-verify:
	$(PY) scripts/business_case.py --verify

# The per-transaction export the live demo page runs on. Refuses to write unless
# the exported rows rebuild the committed metrics and business case. Needs
# data/creditcard.csv.
demo-data:
	$(PY) scripts/demo_data.py

# Provenance of the demo export: retrain on the real data and require the
# committed file byte for byte (also step 3 of `make reproduce`). Needs
# data/creditcard.csv.
demo-verify:
	$(PY) scripts/demo_data.py --verify

# Load the demo page in headless Chromium and check what it displays against
# the committed artifacts. Needs playwright + `python -m playwright install chromium`.
site-check:
	$(PY) scripts/check_site.py

serve:
	$(PY) -m uvicorn ledgersentry.service:app --app-dir src

dashboard:
	$(PY) -m streamlit run dashboard/app.py

# Deploy to a throwaway kind cluster, smoke + load test, rolling restart under
# load (needs docker, kind, kubectl). Same script the k8s CI workflow runs.
k8s-e2e:
	bash scripts/k8s_e2e.sh

# terraform apply deploy/terraform/kubernetes to kind, re-plan, parity, destroy.
tf-kind:
	bash scripts/tf_kind_e2e.sh

# MLflow + Evidently. Once installed, train/compare/compare-boosters log runs to
# a local store (mlflow.db + mlruns/, gitignored); LEDGERSENTRY_MLFLOW=0 turns it off.
install-mlops:
	$(PY) -m pip install -r requirements-mlops.txt

mlflow-ui:
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db

# Evidently drift + classification report, inner-validation window vs the test
# fold (both time windows of the committed split) -> reports/evidently_<source>.html
drift-report:
	$(PY) scripts/evidently_report.py
	$(PY) scripts/check_evidently_html.py

# kubeconform -strict on the rendered base, kind overlay and k6 Job (needs kubectl).
k8s-schema:
	bash scripts/k8s_schema.sh

# docker compose up, smoke-test API and dashboard, down (needs docker).
compose-smoke:
	bash scripts/compose_smoke.sh
