import polars as pl

from animeippo.recommendation import dataset

import tests.test_data as test_data


def test_recommendations_can_be_cached_to_lru_cache():
    recommendations = pl.DataFrame(test_data.FORMATTED_ANI_SEASONAL_LIST)

    dset = dataset.RecommendationModel(None, None)
    dset.recommendations = recommendations

    actual = dset.recommendations_explode_cached("genres").item(0, "genres")

    assert actual == recommendations.explode("genres").item(0, "genres")
    assert type(actual) == str