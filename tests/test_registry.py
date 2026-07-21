"""
The registry's job: a new classifier drops in without editing FraudDetector.
The proof is test_registered_model_trains_without_touching_dispatch, which
registers a model that did not exist when model.py was written and runs it
through the full pipeline untouched.
"""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from ledgersentry import data, registry
from ledgersentry.model import FraudDetector


def _split_synthetic(seed=11):
    raw = data.make_synthetic(n_rows=3000, fraud_rate=0.03, seed=seed)
    df = data.engineer_time_features(raw)
    train_df, test_df = data.temporal_grouped_split(df, test_size=0.25)
    pre = data.build_preprocessor(train_df)
    X_train = pre.fit_transform(train_df)
    X_test = pre.transform(test_df)
    return X_train, train_df["is_fraud"].to_numpy(), X_test, test_df["is_fraud"].to_numpy()


def test_all_shipped_models_available():
    assert {
        "hist_gbdt", "hist_gbdt_shallow", "hist_gbdt_deep", "logreg"
    } <= set(registry.available())


def test_unknown_model_is_a_clear_error():
    with pytest.raises(ValueError, match="unknown model"):
        registry.create("no_such_model", 42, 200, 0.1)
    with pytest.raises(ValueError, match="hist_gbdt"):
        # the error must NAME what is available
        registry.create("no_such_model", 42, 200, 0.1)


def test_reregistering_a_name_is_a_hard_error():
    with pytest.raises(ValueError, match="already registered"):

        @registry.register("hist_gbdt")
        def _shadow(random_state, max_iter, learning_rate):  # pragma: no cover
            return None


@pytest.mark.parametrize(
    "model_name", ["hist_gbdt", "hist_gbdt_shallow", "hist_gbdt_deep", "logreg"]
)
def test_models_swap_through_one_interface(model_name):
    """Same pipeline, same reject knob, different classifier - each must learn
    real signal (beat the no-skill baseline) and honor the abstain contract."""
    X_train, y_train, X_test, y_test = _split_synthetic()
    det = FraudDetector(max_iter=50, model=model_name).fit(X_train, y_train)
    p = det.predict_proba_fraud(X_test)
    assert ((p >= 0) & (p <= 1)).all()
    assert average_precision_score(y_test, p) > y_test.mean()
    decision, _, _ = det.decide(X_test, review_threshold=0.999)
    assert set(np.unique(decision)) <= {"fraud", "legit", "review"}


def test_logreg_survives_nan_inputs():
    """The serving path turns missing fields into NaN; the linear baseline is
    imputer-wrapped so it must not crash on them (hist_gbdt handles NaN natively)."""
    X_train, y_train, X_test, _ = _split_synthetic()
    det = FraudDetector(max_iter=50, model="logreg").fit(X_train, y_train)
    X_nan = np.asarray(X_test, dtype=float).copy()
    X_nan[0, 0] = np.nan
    p = det.predict_proba_fraud(X_nan[:5])
    assert ((p >= 0) & (p <= 1)).all()


def test_registered_model_trains_without_touching_dispatch():
    """Register a classifier the core has never heard of; FraudDetector must
    train and serve it with zero edits to model.py."""
    from sklearn.tree import DecisionTreeClassifier

    name = "test_only_tree"
    if name not in registry.available():  # guard against test reruns in-process

        @registry.register(name)
        def _tree(random_state, max_iter, learning_rate):
            return DecisionTreeClassifier(random_state=random_state, max_depth=4)

    X_train, y_train, X_test, y_test = _split_synthetic()
    det = FraudDetector(model=name).fit(X_train, y_train)
    p = det.predict_proba_fraud(X_test)
    assert len(p) == len(y_test)
    assert ((p >= 0) & (p <= 1)).all()


def test_default_model_is_unchanged():
    """The committed metrics were produced by hist_gbdt with these exact
    defaults; the registry refactor must not have moved them."""
    det = FraudDetector()
    assert det.model == "hist_gbdt"
    assert (det.random_state, det.max_iter, det.learning_rate) == (42, 200, 0.1)
