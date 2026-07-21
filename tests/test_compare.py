import numpy as np
import pandas as pd

from ledgersentry import velocity
from ledgersentry.compare import CONFIGS, FEATURE_SETS, paired_delta_bootstrap


def _scores(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    noise = rng.normal(0, 0.3, size=n)
    good = np.clip(y * 0.7 + 0.1 + noise * 0.3, 0, 1)
    weak = np.clip(y * 0.2 + 0.4 + noise, 0, 1)
    return y, weak, good


def test_delta_is_positive_when_the_challenger_ranks_better():
    y, weak, good = _scores()
    d = paired_delta_bootstrap(y, weak, good, n_resamples=200, seed=1)
    assert d["delta_pr_auc"] > 0
    assert d["share_of_resamples_challenger_wins"] > 0.9
    assert d["ci_lower"] > 0
    assert d["interval_excludes_zero"]


def test_delta_is_negative_when_the_challenger_ranks_worse():
    y, weak, good = _scores()
    d = paired_delta_bootstrap(y, good, weak, n_resamples=200, seed=1)
    assert d["delta_pr_auc"] < 0
    assert d["ci_upper"] < 0


def test_identical_scores_give_a_zero_delta_and_an_interval_on_zero():
    y, weak, _ = _scores()
    d = paired_delta_bootstrap(y, weak, weak, n_resamples=100, seed=1)
    assert d["delta_pr_auc"] == 0.0
    assert d["ci_lower"] == 0.0 and d["ci_upper"] == 0.0
    assert not d["interval_excludes_zero"]


def test_delta_is_deterministic():
    y, weak, good = _scores()
    a = paired_delta_bootstrap(y, weak, good, n_resamples=100, seed=7)
    b = paired_delta_bootstrap(y, weak, good, n_resamples=100, seed=7)
    assert a == b


def test_constant_columns_are_reported_as_degenerate():
    df = pd.DataFrame({"f_vel_count_1min": [0.0, 0.0, 0.0], "f_vel_amt_sum_1h": [1.0, 2.0, 3.0]})
    dead = velocity.degenerate_columns(df, ["f_vel_count_1min", "f_vel_amt_sum_1h"])
    assert dead == ["f_vel_count_1min"]


def test_the_incumbent_is_the_first_config_and_the_first_feature_set():
    """The comparison only means something if the committed run is one of the
    rows: gbdt_default on the base features IS the model behind metrics.json."""
    assert FEATURE_SETS[0] == "base"
    assert CONFIGS[0].name == "gbdt_default"
    assert (CONFIGS[0].model, CONFIGS[0].max_iter, CONFIGS[0].learning_rate) == (
        "hist_gbdt", 200, 0.1
    )
