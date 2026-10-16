"""scripts/headline.py: the README's reproducing command for the headline.

Offline. It must rebuild the published numbers from the committed export, pass on
a clean clone, and fail (not pass quietly) when a committed number is wrong."""
import json
import shutil
from pathlib import Path

import pytest

from ledgersentry.headline import format_report, load_artifacts, main, recompute

ART = Path(__file__).resolve().parents[1] / "artifacts"


def test_recompute_matches_the_published_headline():
    demo, metrics, business = load_artifacts(ART)
    r = recompute(demo, business["operating_threshold"])
    assert r["pr_auc"] == metrics["pr_auc"] == 0.7278
    assert r["pr_auc_random_baseline"] == metrics["pr_auc_random_baseline"]
    assert (r["n_test"], r["n_test_fraud"]) == (56961, 75)
    committed = {row["review_threshold"]: row for row in metrics["coverage_precision_curve"]}
    for row in r["knob"]:
        assert row == committed[row["review_threshold"]]
    # the two operating points the README quotes, kept apart on purpose:
    # 84% recall is the fully automated 0.5 cut; 89.8% precision is the 0.95 band
    at = {row["review_threshold"]: row for row in r["knob"]}
    assert at[0.5]["recall_auto"] == 0.84 and at[0.5]["precision_on_flagged"] == 0.2917
    assert at[0.95]["precision_on_flagged"] == 0.8983 and at[0.95]["recall_auto"] == 0.7067


def test_report_prints_the_readme_figures():
    demo, _, business = load_artifacts(ART)
    text = format_report(recompute(demo, business["operating_threshold"]))
    for needle in ("PR-AUC 0.7278", "56,961", "89.8% precise", "709.1", "92% of frauds surfaced"):
        assert needle in text, needle


def test_main_passes_on_the_committed_artifacts(capsys):
    assert main([]) == 0
    assert "PASS" in capsys.readouterr().out


def test_main_fails_when_a_committed_number_is_edited(tmp_path, capsys):
    for name in ("demo_scores", "metrics", "business_case"):
        shutil.copy(ART / f"{name}_ulb_creditcard.json", tmp_path)
    m = tmp_path / "metrics_ulb_creditcard.json"
    payload = json.loads(m.read_text())
    payload["pr_auc"] = 0.80  # a flattering hand edit
    m.write_text(json.dumps(payload))
    assert main(["--artifacts", str(tmp_path)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_missing_artifact_is_an_error_not_a_skip(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_artifacts(tmp_path)
