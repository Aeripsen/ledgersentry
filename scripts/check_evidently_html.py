"""Check that the committed Evidently HTML says what the committed summary says.
Run:  python scripts/check_evidently_html.py [artifacts/evidently_summary_<source>.json]
      (stdlib only; evidently not needed; default: the ulb_creditcard summary)

The HTML report is what GitHub Pages serves, but it is not byte-reproducible
(every widget gets a fresh random id), so CI cannot git-diff it the way it diffs
artifacts/evidently_summary_<source>.json. Instead this decodes the report data Evidently
embeds in the page (one JSON object assigned to `var metric_<id>`) and compares it
with the summary, value by value:

  * the column counters (columns checked, drifted columns, share drifted)
  * Evidently's dataset-drift line against the summary's dataset_drift
  * every per-column drift row: the set marked Detected must be exactly the
    summary's drifted set, with the same stattest and score
  * the Model Quality counters of both windows (threshold 0.5) against the
    summary's accuracy, precision, recall, F1, ROC AUC and log loss

The page prints its numbers rounded (scores to 6 places, quality counters to 3),
so a value matches when it is within that rounding of the summary's. Exit 1 on
any mismatch, so a stale page cannot sit next to a summary that moved.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "artifacts" / "evidently_summary_ulb_creditcard.json"
# summary performance key -> the label on Evidently's Model Quality counter
QUALITY = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
           "f1": "F1", "roc_auc": "ROC AUC", "log_loss": "LogLoss"}
WINDOWS = {"at_threshold_0.5_current": "Current: Model Quality Metrics",
           "at_threshold_0.5_reference": "Reference: Model Quality Metrics"}
SCORE_TOL = 0.5e-4 + 1e-6    # summary rounds scores to 4 places, the page to 6
QUALITY_TOL = 0.5e-3 + 0.5e-4  # the page prints 3 places, the summary 4


def extract(html_path: Path) -> dict[str, Any]:
    """The counters (by widget title) and the per-column drift rows in the page."""
    text = html_path.read_text(encoding="utf-8")
    m = re.search(r"var metric_[0-9a-f]+ = ", text)
    if m is None:
        raise ValueError(f"{html_path}: no embedded Evidently report data")
    report, _ = json.JSONDecoder().raw_decode(text, m.end())
    counters: dict[str, dict[str, str]] = {}
    columns: dict[str, dict[str, Any]] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "column_name" in node and "drift_score" in node:
                columns[node["column_name"]] = {"stattest": node.get("stattest_name"),
                                                "score": float(node["drift_score"]),
                                                "drifted": node.get("data_drift") == "Detected"}
            params = node.get("params")
            if node.get("type") == "counter" and isinstance(params, dict):
                bucket = counters.setdefault(node.get("title", ""), {})
                for c in params.get("counters", []):
                    bucket[c["label"]] = c["value"]
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(report)
    return {"counters": counters, "columns": columns}


def compare(summary: dict[str, Any], page: dict[str, Any]) -> list[str]:
    bad: list[str] = []
    top = page["counters"].get("", {})
    cols = page["columns"]
    d = summary["data_drift"]
    n = d["n_columns"]

    def expect(what: str, got: Any, want: Any) -> None:
        if got != want:
            bad.append(f"{what}: page {got!r}, summary {want!r}")

    expect("columns counter", top.get("Columns"), str(n))
    expect("per-column rows", len(cols), n)
    expect("drifted columns counter", top.get("Drifted Columns"), str(d["drifted_columns"]))
    expect("columns marked Detected", sorted(c for c, r in cols.items() if r["drifted"]),
           sorted(d["drifted"]))
    share = top.get("Share of Drifted Columns")
    if share is None or abs(float(share) - d["drifted_share"]) > SCORE_TOL:
        bad.append(f"share drifted: page {share!r}, summary {d['drifted_share']!r}")
    verdict = [k for k in top if k.startswith("Dataset Drift is")]
    page_drift = bool(verdict) and not verdict[0].startswith("Dataset Drift is NOT")
    if not verdict:
        bad.append("dataset drift line missing from the page")
    expect("dataset drift", page_drift, d["dataset_drift"])
    for col, r in d["drifted"].items():
        c = cols.get(col)
        if c is None:
            bad.append(f"{col}: not in the page")
            continue
        expect(f"{col} stattest", c["stattest"], r["method"])
        if abs(c["score"] - r["score"]) > SCORE_TOL:
            bad.append(f"{col} score: page {c['score']}, summary {r['score']}")
    for window, title in WINDOWS.items():
        shown = page["counters"].get(title)
        if shown is None:
            bad.append(f"{title}: missing from the page")
            continue
        for key, label in QUALITY.items():
            want = summary["performance"][window][key]
            got = shown.get(label)
            if got is None or abs(float(got) - want) > QUALITY_TOL:
                bad.append(f"{title} {label}: page {got!r}, summary {want!r}")
    return bad


def main(summary_path: Path = SUMMARY) -> int:
    summary = json.loads(summary_path.read_text())
    html = ROOT / summary["html_report"]
    bad = compare(summary, extract(html))
    for b in bad:
        print(f"[check] MISMATCH {b}")
    if bad:
        print(f"[check] FAIL: {html.name} does not match {summary_path.name}")
        return 1
    print(f"[check] {html.name} matches {summary_path.name}: counters, dataset drift, "
          f"per-column drift rows, and both windows' Model Quality counters")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else SUMMARY))
