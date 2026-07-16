"""
Model registry: adding a classifier is one register() call, not an edit to
FraudDetector's dispatch.

Two implementations ship today, both CI-tested through the full pipeline:

  hist_gbdt  HistGradientBoostingClassifier - the default and the model behind
             every committed number. Chosen because it ships inside scikit-learn
             (no compiled-wheel risk in CI) and handles missing values natively,
             which the serving path relies on (missing fields become NaN).
  logreg     scaled logistic regression - the honest linear baseline. Any
             gradient-boosted result should be read against it: if boosting
             cannot beat a linear model, the features are the problem. Wrapped
             with median imputation so it survives the same NaN inputs serving
             produces (a documented baseline compromise; hist_gbdt needs no
             imputation and that is one reason it is the default).

xgboost/lightgbm are deliberately NOT vendored as optional dependencies: a
registry entry for a package the repo neither installs nor tests would be dead
code. Dropping one in is three lines in your own code:

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
