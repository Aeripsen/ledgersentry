# Convenience targets. Every one is a thin wrapper over a plain command that is
# also documented in the README, so Windows users without make lose nothing.

PY ?= python

.PHONY: install test lint train reproduce bench bootstrap compare cost serve dashboard

install:
	$(PY) -m pip install -r requirements.txt

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

# Expected cost per review threshold on calibrated scores, priced under
# several illustrative cost triples so the optimum's dependence on the
# assumptions is visible.
cost:
	$(PY) scripts/cost.py

serve:
	$(PY) -m uvicorn ledgersentry.service:app --app-dir src

dashboard:
	$(PY) -m streamlit run dashboard/app.py
