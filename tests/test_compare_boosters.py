from ledgersentry.compare_boosters import CONFIGS


def test_the_incumbent_is_the_first_config():
    """The head-to-head only means something if the committed run is one of the
    rows: gbdt_default IS the model behind metrics.json, same convention (and
    same pin) as compare.py's config table."""
    assert CONFIGS[0].name == "gbdt_default"
    assert (CONFIGS[0].model, CONFIGS[0].max_iter, CONFIGS[0].learning_rate) == (
        "hist_gbdt", 200, 0.1
    )


def test_the_default_challenger_matches_the_incumbent_budget():
    """lgbm_default must mirror the incumbent's boosting budget exactly, or the
    headline delta measures budget differences and gets read as a library
    difference."""
    by_name = {c.name: c for c in CONFIGS}
    lgbm = by_name["lgbm_default"]
    assert lgbm.model == "lgbm"
    assert (lgbm.max_iter, lgbm.learning_rate) == (CONFIGS[0].max_iter, CONFIGS[0].learning_rate)


def test_every_challenger_is_lgbm():
    """This pipeline answers one question (library vs library); hist_gbdt
    variants already have their comparison in compare.py and a second copy of
    their numbers under a different filename would drift."""
    assert all(c.model == "lgbm" for c in CONFIGS[1:])
