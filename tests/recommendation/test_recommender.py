import pandas as pd

from animeippo.recommendation import recommender, dataset
from tests.recommendation.test_recommendation import ProviderStub
from tests import test_data

from functools import partial


class EngineStub:
    def fit_predict(self, dataset):
        return dataset.seasonal[::-1]


async def databuilder_stub(h, i, j, k, watchlist=None, seasonal=None):
    return dataset.UserDataSet(watchlist, seasonal)


def test_recommender_can_return_plain_seasonal_data():
    seasonal = pd.DataFrame(test_data.FORMATTED_MAL_SEASONAL_LIST)

    provider = ProviderStub()
    engine = None
    databuilder = partial(databuilder_stub, watchlist=None, seasonal=seasonal)

    rec = recommender.AnimeRecommender(provider, engine, databuilder)
    recommendations = rec.recommend_seasonal_anime("2013", "winter")

    assert seasonal.loc[0]["title"] in recommendations["title"].to_list()


def test_recommender_can_recommend_seasonal_data_for_user():
    seasonal = pd.DataFrame(test_data.FORMATTED_MAL_SEASONAL_LIST)
    watchlist = pd.DataFrame(test_data.FORMATTED_MAL_USER_LIST)

    provider = ProviderStub()
    engine = EngineStub()
    databuilder = partial(databuilder_stub, watchlist=watchlist, seasonal=seasonal)

    rec = recommender.AnimeRecommender(provider, engine, databuilder)
    recommendations = rec.recommend_seasonal_anime("2013", "winter", "Janiskeisari")

    assert seasonal.loc[0]["title"] in recommendations["title"].to_list()
