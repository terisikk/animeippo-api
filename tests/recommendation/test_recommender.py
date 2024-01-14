import polars as pl
import pytest

from animeippo.recommendation import model, recommender, profile
from tests.recommendation.test_engine import ProviderStub
from tests import test_data


class EngineStub:
    def fit_predict(self, dataset):
        return dataset.seasonal[::-1]

    def categorize_anime(self, dataset):
        return [[1, 2, 3]]


async def databuilder_stub(h, i, j, k, watchlist=None, seasonal=None):
    return model.RecommendationModel(profile.UserProfile("Test", watchlist), seasonal)


def test_recommender_can_return_plain_seasonal_data():
    seasonal = pl.DataFrame(test_data.FORMATTED_MAL_SEASONAL_LIST)

    provider = ProviderStub()
    engine = None

    rec = recommender.AnimeRecommender(provider, engine)
    data = rec.recommend_seasonal_anime("2013", "winter")

    assert seasonal.item(0, "title") in data.recommendations["title"].to_list()


def test_recommender_can_recommend_seasonal_data_for_user():
    seasonal = pl.DataFrame(test_data.FORMATTED_MAL_SEASONAL_LIST)

    provider = ProviderStub()
    engine = EngineStub()

    rec = recommender.AnimeRecommender(provider, engine)
    data = rec.recommend_seasonal_anime("2013", "winter", "Janiskeisari")

    assert seasonal.item(0, "title") in data.recommendations["title"].to_list()


def test_recommender_categories():
    provider = ProviderStub()
    engine = EngineStub()

    rec = recommender.AnimeRecommender(provider, engine)
    data = rec.recommend_seasonal_anime("2013", "winter", "Janiskeisari")
    categories = rec.get_categories(data)

    assert len(categories) > 0
    assert categories == [[1, 2, 3]]


@pytest.mark.asyncio
async def test_recommender_can_get_data_when_async_loop_is_already_running():
    seasonal = pl.DataFrame(test_data.FORMATTED_MAL_SEASONAL_LIST)

    provider = ProviderStub()
    engine = EngineStub()

    rec = recommender.AnimeRecommender(provider, engine)
    data = rec.recommend_seasonal_anime("2013", "winter", "Janiskeisari")

    assert seasonal.item(0, "title") in data.recommendations["title"].to_list()
