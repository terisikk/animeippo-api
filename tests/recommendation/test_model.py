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
