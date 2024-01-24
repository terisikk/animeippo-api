import polars as pl
import pytest

from animeippo.profiling.model import UserProfile
from animeippo.recommendation import recommender
from animeippo.recommendation.model import RecommendationModel
from tests import test_data
from tests.recommendation.test_engine import ProviderStub


class EngineStub:
    def fit_predict(self, dataset):
        return dataset.seasonal[::-1]

    def categorize_anime(self, dataset):
        return [[1, 2, 3]]


def test_recommender_can_return_plain_seasonal_data():
    seasonal = pl.DataFrame(test_data.FORMATTED_MAL_SEASONAL_LIST)

    provider = ProviderStub()
    engine = None

    rec = recommender.AnimeRecommender(
        provider=provider,
        engine=engine,
        recommendation_model_cls=RecommendationModel,
        profile_model_cls=UserProfile,
    )
    data = rec.recommend_seasonal_anime("2013", "winter")

    assert seasonal.item(0, "title") in data.recommendations["title"].to_list()


def test_recommender_can_recommend_seasonal_data_for_user():
    seasonal = pl.DataFrame(test_data.FORMATTED_MAL_SEASONAL_LIST)

    provider = ProviderStub()
    engine = EngineStub()

    rec = recommender.AnimeRecommender(
        provider=provider,
        engine=engine,
        recommendation_model_cls=RecommendationModel,
        profile_model_cls=UserProfile,
    )
    data = rec.recommend_seasonal_anime("2013", "winter", "Janiskeisari")

    assert seasonal.item(0, "title") in data.recommendations["title"].to_list()


def test_recommender_can_fetch_related_anime_when_needed():
    provider = ProviderStub()
    engine = EngineStub()

    rec = recommender.AnimeRecommender(
        provider=provider,
        engine=engine,
        recommendation_model_cls=RecommendationModel,
        profile_model_cls=UserProfile,
        fetch_related_anime=True,
    )
    data = rec.recommend_seasonal_anime("2013", "winter", "Janiskeisari")

    assert data.seasonal["continuation_to"].to_list()[0] == [1]


def test_recommender_categories():
    provider = ProviderStub()
    engine = EngineStub()

    rec = recommender.AnimeRecommender(
        provider=provider,
        engine=engine,
        recommendation_model_cls=RecommendationModel,
        profile_model_cls=UserProfile,
    )
    data = rec.recommend_seasonal_anime("2013", "winter", "Janiskeisari")
    categories = rec.get_categories(data)

    assert len(categories) > 0
    assert categories == [[1, 2, 3]]


@pytest.mark.asyncio
async def test_recommender_can_get_data_when_async_loop_is_already_running():
    seasonal = pl.DataFrame(test_data.FORMATTED_MAL_SEASONAL_LIST)

    provider = ProviderStub()
    engine = EngineStub()

    rec = recommender.AnimeRecommender(
        provider=provider,
        engine=engine,
        recommendation_model_cls=RecommendationModel,
        profile_model_cls=UserProfile,
    )
    data = rec.recommend_seasonal_anime("2013", "winter", "Janiskeisari")

    assert seasonal.item(0, "title") in data.recommendations["title"].to_list()
