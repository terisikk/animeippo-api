import polars as pl

import tests.test_data as test_data
from animeippo.recommendation import model


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

    dset = model.RecommendationModel(None, seasonal)
    dset.watchlist = watchlist

    dset.filter_continuation()

    assert dset.seasonal["continuation_to"].dtype == pl.List(pl.Int64)
    assert dset.seasonal["continuation_to"].to_list()[0] == [30]


def test_features_can_be_extracted_from_ranks():
    seasonal = pl.DataFrame(test_data.FORMATTED_ANI_SEASONAL_LIST)
    watchlist = pl.DataFrame(test_data.FORMATTED_ANI_USER_LIST)

    dset = model.RecommendationModel(None, seasonal)
    dset.watchlist = watchlist

    dset.all_features = dset.extract_features()

    assert dset.all_features is not None
