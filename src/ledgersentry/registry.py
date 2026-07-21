"""
Model registry: adding a classifier is one register() call, not an edit to
FraudDetector's dispatch.

Four implementations ship today, all CI-tested through the full pipeline:

  hist_gbdt  HistGradientBoostingClassifier - the default and the model behind
             every committed number. Chosen because it ships inside scikit-learn
             (no compiled-wheel risk in CI) and handles missing values natively,
             which the serving path relies on (missing fields become NaN).
  hist_gbdt_shallow / hist_gbdt_deep
             the same estimator with regularization dialed down and up, so the
             default is a measured choice instead of an assumed one. Their
             numbers on the real fold are in artifacts/comparison_*.json.
  logreg     scaled logistic regression - the honest linear baseline. Any
             gradient-boosted result should be read against it: if boosting
             cannot beat a linear model, the features are the problem. Wrapped
             with median imputation so it survives the same NaN inputs serving
             produces (a documented baseline compromise; hist_gbdt needs no
             imputation and that is one reason it is the default).

One optional entry ships beside them:

  lgbm       LightGBM's LGBMClassifier, registered so scripts/compare_boosters.py
             can measure the "would a dedicated boosting library beat the
             sklearn default?" question instead of leaving it asserted. The
             import is deferred into the factory: the registry, the core
             pipeline, and CI all work without lightgbm installed, and asking
             for "lgbm" without it is a clear error naming the fix
             (requirements-analysis.txt). The cost of registering it is that
             the registry now names a model the base install cannot create;
             the docstring on the factory and the error message carry that.

xgboost stays unvendored - one measured challenger from the same model family
answers the question, and a second compiled dependency would buy a second
number, not a second insight. Dropping it in is still three lines in your own
code:

    from ledgersentry.registry import register
    @register("xgboost")
    def _xgboost(random_state: int, max_iter: int, learning_rate: float):
        return XGBClassifier(n_estimators=max_iter, learning_rate=learning_rate,
                             random_state=random_state)

Every factory takes the same three hyperparameters (the ones the config file
exposes) and is free to ignore what does not apply to it. Every returned
estimator must support fit(X, y, sample_weight=...) (Pipelines get the weight
routed to their final step by FraudDetector) and predict_proba.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ModelFactory = Callable[[int, int, float], Any]

_REGISTRY: dict[str, ModelFactory] = {}


def register(name: str) -> Callable[[ModelFactory], ModelFactory]:
    """Register a model factory under `name`. Re-registering a taken name is a
    hard error: silently shadowing the model behind committed metrics is exactly
    the kind of quiet swap this repo exists to prevent."""

    def deco(factory: ModelFactory) -> ModelFactory:
        if name in _REGISTRY:
            raise ValueError(f"model {name!r} is already registered")
        _REGISTRY[name] = factory
        return factory

    return deco


def create(name: str, random_state: int, max_iter: int, learning_rate: float) -> Any:
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown model {name!r}; registered: {sorted(_REGISTRY)}"
        ) from None
    return factory(random_state, max_iter, learning_rate)


def available() -> list[str]:
    return sorted(_REGISTRY)


@register("hist_gbdt")
def _hist_gbdt(random_state: int, max_iter: int, learning_rate: float) -> Any:
    return HistGradientBoostingClassifier(
        random_state=random_state, max_iter=max_iter, learning_rate=learning_rate
    )


@register("hist_gbdt_shallow")
def _hist_gbdt_shallow(random_state: int, max_iter: int, learning_rate: float) -> Any:
    # Heavily regularized boosting: the train split has 417 frauds, so the
    # default 31-leaf trees have enough capacity to carve out individual
    # positives. Fewer leaves, a bigger leaf floor and an L2 penalty are the
    # standard answer. Registered so scripts/compare.py can measure whether
    # that argument survives contact with the holdout.
    return HistGradientBoostingClassifier(
        random_state=random_state,
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=8,
        min_samples_leaf=50,
        l2_regularization=1.0,
    )


@register("hist_gbdt_deep")
def _hist_gbdt_deep(random_state: int, max_iter: int, learning_rate: float) -> Any:
    # The other direction from shallow, so the comparison has a range and not
    # just one alternative next to the default.
    return HistGradientBoostingClassifier(
        random_state=random_state,
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=63,
        min_samples_leaf=10,
        l2_regularization=0.0,
    )


@register("lgbm")
def _lgbm(random_state: int, max_iter: int, learning_rate: float) -> Any:
    # Optional dependency, imported at creation time on purpose: the registry
    # must stay importable (and CI green) on the base install, and the price is
    # that a bad environment surfaces here, at create(), not at import. The
    # error names the fix so that trade stays cheap.
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError(
            "model 'lgbm' needs the optional lightgbm package: "
            "pip install -r requirements-analysis.txt (or: make install-analysis)"
        ) from exc
    # num_leaves=31 is LightGBM's own default and equals hist_gbdt's
    # max_leaf_nodes default, so the head-to-head in compare_boosters.py is
    # library vs library at matched capacity, not a hidden tuning advantage.
    # deterministic + force_row_wise pin the histogram construction order;
    # without them LightGBM may pick a layout per run and the committed
    # comparison numbers would not reproduce.
    return LGBMClassifier(
        n_estimators=max_iter,
        learning_rate=learning_rate,
        random_state=random_state,
        num_leaves=31,
        deterministic=True,
        force_row_wise=True,
        verbosity=-1,
    )


@register("logreg")
def _logreg(random_state: int, max_iter: int, learning_rate: float) -> Any:
    # learning_rate does not apply to logistic regression and is ignored.
    # max_iter here is the solver's iteration cap, floored so the default
    # config (tuned for boosting rounds) still converges.
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    random_state=random_state, max_iter=max(1000, max_iter)
                ),
            ),
        ]
    )
