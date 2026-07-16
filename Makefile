# Convenience targets. Every one is a thin wrapper over a plain command that is
# also documented in the README, so Windows users without make lose nothing.

PY ?= python

.PHONY: install test lint train reproduce bench serve dashboard

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

serve:
	$(PY) -m uvicorn ledgersentry.service:app --app-dir src

dashboard:
	$(PY) -m streamlit run dashboard/app.py
