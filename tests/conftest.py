"""Suite-wide: never log to the developer's MLflow store. Some tests train on the
synthetic fixture, and with mlflow installed train.py would record those runs
next to the real ones."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_mlflow_logging(monkeypatch):
    monkeypatch.setenv("LEDGERSENTRY_MLFLOW", "0")
