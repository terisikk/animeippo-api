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
