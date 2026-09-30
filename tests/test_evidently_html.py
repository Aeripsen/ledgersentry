"""The committed Evidently page must say what the committed summary says.
Stdlib only: runs in the base CI job without evidently installed."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_evidently_html",
                                              ROOT / "scripts" / "check_evidently_html.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
SUMMARY = json.loads(checker.SUMMARY.read_text())
PAGE = checker.extract(ROOT / SUMMARY["html_report"])


def test_committed_page_matches_committed_summary():
    assert checker.compare(SUMMARY, PAGE) == []


def test_page_carries_every_checked_column():
    assert len(PAGE["columns"]) == SUMMARY["data_drift"]["n_columns"]


def test_a_summary_that_moved_is_caught():
    moved = copy.deepcopy(SUMMARY)
    col = next(iter(moved["data_drift"]["drifted"]))
    moved["data_drift"]["drifted"][col]["score"] += 0.001
    moved["performance"]["at_threshold_0.5_current"]["recall"] += 0.01
    bad = checker.compare(moved, PAGE)
    assert any(col in b for b in bad) and any("Recall" in b for b in bad)
