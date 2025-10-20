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


class SimpleProviderStub:
    """Provider stub without async context manager methods."""

    def __init__(self):
        pass

    async def get_seasonal_anime_list(self, *args, **kwargs):
        return pl.DataFrame(test_data.FORMATTED_MAL_SEASONAL_LIST)

    async def get_user_anime_list(self, *args, **kwargs):
        return pl.DataFrame(test_data.FORMATTED_MAL_USER_LIST)

    async def get_user_manga_list(self, *args, **kwargs):
        return pl.DataFrame()

    def get_nsfw_tags(self):
        return []


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


@pytest.mark.asyncio
async def test_recommender_context_manager():
    """Test that AnimeRecommender properly works as async context manager."""
    provider = ProviderStub()
    engine = EngineStub()

    async with recommender.AnimeRecommender(
        provider=provider,
        engine=engine,
        recommendation_model_cls=RecommendationModel,
        profile_model_cls=UserProfile,
    ) as rec:
        # Recommender should work normally inside context
        assert rec.provider is not None
        assert rec.engine is not None

    # After exiting context, if provider has cleanup, it should be called
    # (ProviderStub doesn't have __aenter__/__aexit__, so this tests the hasattr branch)


@pytest.mark.asyncio
async def test_recommender_context_manager_with_provider_cleanup():
    """Test that AnimeRecommender delegates context manager to provider when available."""

    class ProviderWithCleanup:
        def __init__(self):
            self.entered = False
            self.exited = False

        async def __aenter__(self):
            self.entered = True
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            self.exited = True
            return False

        async def get_seasonal_anime_list(self, *args, **kwargs):
            return pl.DataFrame(test_data.FORMATTED_MAL_SEASONAL_LIST)

        async def get_user_anime_list(self, *args, **kwargs):
            return pl.DataFrame(test_data.FORMATTED_MAL_USER_LIST)

        async def get_user_manga_list(self, *args, **kwargs):
            return pl.DataFrame(test_data.FORMATTED_MAL_USER_LIST)

        def get_nsfw_tags(self):
            return []

    provider = ProviderWithCleanup()
    engine = EngineStub()

    async with recommender.AnimeRecommender(
        provider=provider,
        engine=engine,
        recommendation_model_cls=RecommendationModel,
        profile_model_cls=UserProfile,
    ):
        # Provider's __aenter__ should have been called
        assert provider.entered is True
        assert provider.exited is False

    # After exiting, provider's __aexit__ should have been called
    assert provider.exited is True


@pytest.mark.asyncio
async def test_recommender_context_manager_without_provider_context_manager():
    """Test that AnimeRecommender works when provider lacks async context manager methods.

    This covers the case for MyAnimeListProvider and MixedProvider which don't
    implement __aenter__ and __aexit__.
    """
    provider = SimpleProviderStub()
    engine = EngineStub()

    # Should work even though provider doesn't have __aenter__/__aexit__
    async with recommender.AnimeRecommender(
        provider=provider,
        engine=engine,
        recommendation_model_cls=RecommendationModel,
        profile_model_cls=UserProfile,
    ) as rec:
        assert rec.provider is not None
        assert rec.engine is not None

    # No errors should occur - the hasattr checks should handle this gracefully
