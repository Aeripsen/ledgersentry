"""The shap pipeline itself needs the optional shap install and the real
artifact, so CI covers the pure pieces: the name cleanup and the ranking
math, which are where a wrong number would actually come from."""
import numpy as np

from ledgersentry.shap_report import display_names, ranking


def test_display_names_strip_only_the_transformer_prefix():
    raw = ["num__amount", "cat__category_travel", "num__f_V1", "plain"]
    assert display_names(raw) == ["amount", "category_travel", "f_V1", "plain"]


def test_display_names_keep_double_underscores_inside_a_feature_name():
    # only the FIRST separator is the ColumnTransformer's; the rest is the name
    assert display_names(["num__f_weird__col"]) == ["f_weird__col"]


def test_ranking_orders_by_mean_abs_not_signed_mean():
    """A feature that pushes half the rows up and half down has a signed mean
    near zero but is doing real work; |SHAP| must rank it, the signed mean must
    expose it."""
    sv = np.array([[2.0, 0.5], [-2.0, 0.5]])
    rows = ranking(sv, ["both_ways", "small_up"], top_k=2)
    assert [r["feature"] for r in rows] == ["both_ways", "small_up"]
    assert rows[0]["mean_abs_shap"] == 2.0
    assert rows[0]["mean_shap_signed"] == 0.0
    assert rows[1]["mean_shap_signed"] == 0.5


def test_ranking_respects_top_k_and_ranks_from_one():
    sv = np.array([[3.0, 1.0, 2.0]])
    rows = ranking(sv, ["a", "b", "c"], top_k=2)
    assert [(r["rank"], r["feature"]) for r in rows] == [(1, "a"), (2, "c")]


def test_ranking_values_are_plain_floats_for_json():
    # np.float64 would serialize, but the artifact contract is plain JSON types
    rows = ranking(np.array([[1.5]]), ["a"], top_k=1)
    assert type(rows[0]["mean_abs_shap"]) is float
    assert type(rows[0]["mean_shap_signed"]) is float
