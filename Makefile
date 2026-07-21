# Convenience targets. Every one is a thin wrapper over a plain command that is
# also documented in the README, so Windows users without make lose nothing.

PY ?= python

.PHONY: install install-analysis test lint train reproduce bench bootstrap compare compare-boosters shap cost serve dashboard

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

serve:
	$(PY) -m uvicorn ledgersentry.service:app --app-dir src

dashboard:
	$(PY) -m streamlit run dashboard/app.py
