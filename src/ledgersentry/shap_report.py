"""
Which features carry the shipped model's fraud signal - SHAP on the exact
artifact behind the committed numbers, not a retrain.

Model card limitation 9 says this repo has no explainability surface at all.
This module is the first piece of one: exact TreeExplainer attributions for the
committed artifacts/ledgersentry.joblib, computed on the same held-out test
fold every other committed number is measured on. Loading the artifact instead
of refitting matters: a refit that drifted by one tree would produce
attributions for a model nobody shipped, and nothing would catch it. The
script cross-checks the artifact's recorded data source against the data on
disk and refuses to explain a mismatch.

What the numbers are: TreeExplainer decomposes each prediction's raw score
into per-feature contributions that sum exactly back to it (checked, and the
max reconstruction error is committed in the artifact). For
HistGradientBoostingClassifier the raw score is the LOG-ODDS margin, not a
probability - a mean |SHAP| of 1.0 means the feature typically moves the
fraud log-odds by about 1, and there is no honest per-feature translation to
probability points because the sigmoid's slope depends on where the other
features already put the score.

Determinism: there is nothing to seed. The artifact's trees are frozen, the
temporal split is order-based with no RNG, every row of the test fold is
explained (no sampling), and TreeExplainer's tree traversal is exact - so two
runs on the same artifact and data are identical. The config's random_state
is recorded in the artifact anyway because it identifies the fold.

The numba stub, in full, because it is the one ugly thing here: shap's wheel
imports numba at package import time for its jitted explainers, and every
numba release as of this writing caps numpy < 2.5, while this repo pins
numpy==2.5.1 (requirements.txt) because `make reproduce` checks the committed
metrics byte for byte and a numpy downgrade would silently break that
guarantee. The only shap code path this module uses is TreeExplainer, whose
tree traversal runs in shap's own compiled C extension and never calls a
numba-jitted function. So when the real numba is absent, _ensure_numba() puts
a do-nothing decorator module in its place, which satisfies shap's import-time
decoration and nothing else. The cost is real and stated: any shap feature
that genuinely executes jitted code (Exact/Partition explainers, the image
maskers) would run interpreted or break in this environment. None of it is
used here, and if the real numba is installed the stub steps aside.

Run: python scripts/shap_report.py
  -> artifacts/shap_<source>.json + artifacts/shap_summary_<source>.png
(needs the optional install: make install-analysis)
"""
from __future__ import annotations

import json
import platform
import sys
import types
from datetime import UTC, datetime
from typing import Any

import numpy as np


def display_names(raw_names: list[str]) -> list[str]:
    """ColumnTransformer prefixes every output column with its transformer name
    ("num__amount", "cat__category_travel"). The prefix says which branch of
    the preprocessor produced the column - true, but noise in a feature
    ranking - so it is stripped for display. Only the first separator goes:
    a feature legitimately containing "__" keeps the rest of its name."""
    return [n.split("__", 1)[1] if "__" in n else n for n in raw_names]


def ranking(shap_values: np.ndarray, names: list[str], top_k: int) -> list[dict[str, Any]]:
    """Top-k features by mean |SHAP| over the given rows, with the signed mean
    kept beside it. The signed mean answers the question the magnitude alone
    cannot: does this feature, on these rows, push toward fraud (positive) or
    away from it - and a signed mean near zero under a large magnitude says
    the feature cuts both ways depending on its value."""
    sv = np.asarray(shap_values, dtype=float)
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_k]
    return [
        {
            "rank": i + 1,
            "feature": names[j],
            "mean_abs_shap": round(float(mean_abs[j]), 4),
            "mean_shap_signed": round(float(sv[:, j].mean()), 4),
        }
        for i, j in enumerate(order)
    ]


def _ensure_numba() -> None:
    """Install the import-surface stub described in the module docstring, only
    if the real numba is absent. shap's import-time needs are exactly: `njit`
    (bare and parametrized decorator) and `numba.typed.List`."""
    try:
        import numba  # noqa: F401

        return
    except ImportError:
        pass

    def njit(*args: Any, **kwargs: Any) -> Any:
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def deco(fn: Any) -> Any:
            return fn

        return deco

    numba_stub = types.ModuleType("numba")
    numba_stub.njit = njit  # type: ignore[attr-defined]
    numba_stub.jit = njit  # type: ignore[attr-defined]
    typed = types.ModuleType("numba.typed")
    typed.List = list  # type: ignore[attr-defined]
    numba_stub.typed = typed  # type: ignore[attr-defined]
    sys.modules["numba"] = numba_stub
    sys.modules["numba.typed"] = typed


TOP_K = 15


def main() -> dict:
    # Heavy and optional imports stay inside main so the module (and its two
    # pure helpers above, which the tests import) loads on the base install.
    import joblib
    import pandas as pd
    import sklearn

    from .config import get_settings
    from .data import engineer_time_features, load, temporal_grouped_split

    _ensure_numba()
    import matplotlib

    matplotlib.use("Agg")  # file output only; never require a display
    import matplotlib.pyplot as plt
    import shap

    cfg = get_settings()
    bundle_path = cfg.artifact_dir / "ledgersentry.joblib"
    bundle = joblib.load(bundle_path)
    pre, detector = bundle["preprocessor"], bundle["model"]

    df = load(data_dir=cfg.data_dir)
    source = df.attrs.get("source", "unknown")

    # The artifact records what it was trained on via the metrics file written
    # in the same train.py run. If the data on disk is a different source, the
    # attributions would describe the shipped model scoring the wrong dataset,
    # which is worse than no report - so it is a hard stop, not a warning.
    metrics_path = cfg.artifact_dir / "metrics.json"
    trained_on = json.loads(metrics_path.read_text())["data_source"]
    if trained_on != source:
        raise SystemExit(
            f"artifact was trained on {trained_on!r} (per {metrics_path}) but the "
            f"data on disk is {source!r}; refusing to explain a mismatched model"
        )

    df = engineer_time_features(df)
    _, test_df = temporal_grouped_split(df, test_size=cfg.test_size)
    y_test = test_df["is_fraud"].to_numpy()
    X_test = pre.transform(test_df)
    names = display_names(list(pre.get_feature_names_out()))
    print(
        f"[data ] source={source} explaining all {len(test_df)} test rows "
        f"({int(y_test.sum())} fraud), {len(names)} features"
    )

    explainer = shap.TreeExplainer(detector.model_)
    shap_values = np.asarray(explainer.shap_values(X_test))
    expected_value = float(np.ravel(explainer.expected_value)[0])

    # Exactness check: contributions + base value must rebuild the model's own
    # raw margin. This is what makes the report auditable rather than trusted.
    raw_margin = detector.model_.decision_function(X_test)
    additivity_err = float(np.abs(raw_margin - (expected_value + shap_values.sum(axis=1))).max())
    print(f"[check] additivity max abs error {additivity_err:.2e} (log-odds units)")

    top_all = ranking(shap_values, names, TOP_K)
    # The fraud-only ranking is the fraud-domain question: on the rows that ARE
    # fraud, what did the model actually key on? On a 0.13% base-rate fold the
    # global ranking is dominated by how features behave on legitimate traffic.
    top_fraud = ranking(shap_values[y_test == 1], names, TOP_K)
    for row in top_all[:5]:
        print(
            f"[top  ] {row['rank']:2d}. {row['feature']:12s} "
            f"mean|SHAP| {row['mean_abs_shap']:.4f} signed {row['mean_shap_signed']:+.4f}"
        )

    report = {
        "data_source": source,
        "is_synthetic": source == "synthetic",
        "measured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "shap": shap.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "explained_model": {
            "artifact": bundle_path.name,
            "model": detector.model,
            "random_state": detector.random_state,
            "max_iter": detector.max_iter,
            "learning_rate": detector.learning_rate,
        },
        "explained_rows": {
            "n_rows": int(len(test_df)),
            "n_fraud": int(y_test.sum()),
            "note": (
                "Every row of the same temporal test fold behind the committed "
                "metrics; no sampling, so the report is deterministic."
            ),
        },
        "units": (
            "Raw log-odds margin of HistGradientBoostingClassifier. mean_abs_shap "
            "1.0 means the feature typically moves the fraud log-odds by about 1. "
            "Not probability points; the sigmoid's slope depends on the rest of "
            "the score, so no fixed per-feature translation exists."
        ),
        "expected_value_log_odds": round(expected_value, 4),
        "additivity_max_abs_error": additivity_err,
        f"top_{TOP_K}_by_mean_abs_shap": top_all,
        f"top_{TOP_K}_by_mean_abs_shap_fraud_rows_only": top_fraud,
        "how_to_read_this": [
            "mean_shap_signed near zero under a large mean_abs_shap means the "
            "feature pushes different rows in different directions; it is not "
            "a weak feature.",
            "On ULB the V columns are PCA-anonymized, so this ranking says which "
            "components the model uses, not anything a human can act on. That is "
            "the dataset's limit, not the method's; on a source with named "
            "features the same script produces reason-code raw material.",
            "Attributions describe the shipped model's behavior on this fold. "
            "They are not causal claims about fraud.",
        ],
    }

    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    out_json = cfg.artifact_dir / f"shap_{source}.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"[save ] {out_json}")

    # The beeswarm carries what the tables cannot: for each top feature, how
    # attribution varies with the feature's value across all 57k rows.
    shap.summary_plot(
        shap_values, features=X_test, feature_names=names, max_display=TOP_K, show=False
    )
    fig = plt.gcf()
    fig.suptitle(f"SHAP summary, shipped model on the {source} test fold", fontsize=11)
    fig.tight_layout()
    out_png = cfg.artifact_dir / f"shap_summary_{source}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save ] {out_png}")
    return report


if __name__ == "__main__":
    main()
