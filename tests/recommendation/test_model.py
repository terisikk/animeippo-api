import polars as pl

from animeippo.recommendation import model
from tests import test_data


def test_recommendations_can_be_cached_to_lru_cache():
    recommendations = pl.DataFrame(test_data.FORMATTED_ANI_SEASONAL_LIST)

    dset = model.RecommendationModel(None, None)
    dset.recommendations = recommendations

    actual = dset.recommendations_explode_cached("genres").item(0, "genres")

    assert actual == recommendations.explode("genres").item(0, "genres")
    assert isinstance(actual, str)


def test_continuation_filtering_skips_when_data_missing():
    dset = model.RecommendationModel(None, None)
    dset.filter_continuation()
    assert dset.seasonal is None


def test_continuation_filtering_works():
    seasonal = pl.DataFrame(test_data.FORMATTED_ANI_SEASONAL_LIST)
    watchlist = pl.DataFrame(test_data.FORMATTED_ANI_USER_LIST)

    seasonal = seasonal.with_columns(user_status=pl.lit(None, pl.Utf8))

    dset = model.RecommendationModel(None, seasonal)
    dset.watchlist = watchlist

    dset.filter_continuation()

    assert dset.seasonal["continuation_to"].dtype == pl.List(pl.Int64)
    assert dset.seasonal["continuation_to"].to_list()[0] == [30]


def test_features_can_be_extracted_from_ranks():
    seasonal = pl.DataFrame(
        test_data.FORMATTED_ANI_SEASONAL_LIST,
        schema_overrides={"features": pl.List(pl.Categorical(ordering="lexical"))},
    )
    watchlist = pl.DataFrame(
        test_data.FORMATTED_ANI_USER_LIST,
        schema_overrides={"features": pl.List(pl.Categorical(ordering="lexical"))},
    )

    dset = model.RecommendationModel(None, seasonal)
    dset.watchlist = watchlist

    dset.all_features = dset.extract_features()

    assert dset.all_features is not None


def test_explode_cache_returns_same_object():
    dset = model.RecommendationModel(None, None)
    dset.watchlist = pl.DataFrame({"features": [["A", "B"], ["C"]]})
    dset.recommendations = pl.DataFrame({"features": [["X"]]})
    dset.seasonal = pl.DataFrame({"features": [["Y"]]})

    assert dset.watchlist_explode_cached("features") is dset.watchlist_explode_cached("features")
    assert dset.recommendations_explode_cached("features") is dset.recommendations_explode_cached(
        "features"
    )
    assert dset.seasonal_explode_cached("features") is dset.seasonal_explode_cached("features")


def test_build_relation_context_tags_summaries():
    watchlist = pl.DataFrame(
        {
            "id": [1, 2],
            "franchise_relations": [
                [
                    {"related_id": 10, "relation_type": "SUMMARY"},
                    {"related_id": 11, "relation_type": "SEQUEL"},
                ],
                [{"related_id": 12, "relation_type": "COMPILATION"}],
            ],
        },
        schema={
            "id": pl.UInt32,
            "franchise_relations": pl.List(
                pl.Struct({"related_id": pl.UInt32, "relation_type": pl.Utf8})
            ),
        },
    )
    seasonal = pl.DataFrame({"id": pl.Series([10, 11, 12, 13], dtype=pl.UInt32)})

    dset = model.RecommendationModel(None, seasonal)
    dset.watchlist = watchlist
    dset.build_relation_context()

    result = dict(
        zip(dset.seasonal["id"].to_list(), dset.seasonal["is_summary"].to_list(), strict=True)
    )
    assert result[10] is True  # SUMMARY
    assert result[11] is False  # SEQUEL
    assert result[12] is True  # COMPILATION
    assert result[13] is False  # not related


def test_build_relation_context_handles_missing_column():
    watchlist = pl.DataFrame({"id": [1, 2]})
    seasonal = pl.DataFrame({"id": pl.Series([10], dtype=pl.UInt32)})

    dset = model.RecommendationModel(None, seasonal)
    dset.watchlist = watchlist
    dset.build_relation_context()

    assert dset.seasonal["is_summary"].to_list() == [False]


def test_cluster_names_and_rankings_are_cached():
    watchlist = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "cluster": [0, 0, 1],
            "features": [["Action"], ["Action"], ["Drama"]],
            "score": [8, 9, 7],
            "user_status": ["COMPLETED", "COMPLETED", "CURRENT"],
        }
    )
    recs = pl.DataFrame({"id": [10, 11], "cluster": [0, 1], "cluster_similarity": [0.8, 0.5]})

    dset = model.RecommendationModel(None, None)
    dset.watchlist = watchlist
    dset.recommendations = recs

    names1 = dset.get_cluster_names({}, set())
    names2 = dset.get_cluster_names({}, set())
    assert names1 is names2

    rankings1 = dset.get_cluster_rankings()
    rankings2 = dset.get_cluster_rankings()
    assert rankings1 is rankings2
