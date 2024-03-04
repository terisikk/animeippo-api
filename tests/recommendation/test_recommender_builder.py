import pytest

from animeippo import providers
from animeippo.profiling.model import UserProfile
from animeippo.recommendation import recommender, recommender_builder
from animeippo.recommendation.model import RecommendationModel
from tests import test_provider


class CacheStub:
    def is_available(self):
        return self.is_available


@pytest.mark.asyncio
async def test_Recommenderbuilder_with_anilist():
    b = (
        recommender_builder.RecommenderBuilder()
        .provider(test_provider.AsyncProviderStub())
        .engine("fake")
        .profile_model_class(UserProfile)
        .recommendation_model_class(RecommendationModel)
    )

    actual = b.build()
    data = await actual.get_dataset(2023, "winter", "test")

    assert isinstance(actual, recommender.AnimeRecommender)
    assert actual.provider is not None
    assert actual.databuilder is not None
    assert actual.engine is not None

    assert "Copper Kamuy 4th Season" in data.seasonal["title"].to_list()


@pytest.mark.asyncio
async def test_Recommenderbuilder_with_mal():
    b = (
        recommender_builder.RecommenderBuilder()
        .provider(test_provider.AsyncProviderStub())
        .engine("fake")
    )

    actual = b.build()
    data = await actual.get_dataset(2023, "winter", "test")

    assert isinstance(actual, recommender.AnimeRecommender)
    assert actual.provider is not None
    assert actual.databuilder is not None
    assert actual.engine is not None

    assert "Copper Kamuy 4th Season" in data.seasonal["title"].to_list()


@pytest.mark.asyncio
async def test_mal_databuilder_does_not_fail_with_missing_data():
    b = (
        recommender_builder.RecommenderBuilder()
        .provider(test_provider.FaultyProviderStub())
        .engine("fake")
    )

    actual = b.build()
    data = await actual.get_dataset(2023, "winter", "test")

    assert data.seasonal is None
    assert data.watchlist is None


@pytest.mark.asyncio
async def test_anilist_databuilder_does_not_fail_with_missing_data():
    b = (
        recommender_builder.RecommenderBuilder()
        .provider(test_provider.FaultyProviderStub())
        .engine("fake")
    )

    actual = b.build()
    data = await actual.get_dataset(2023, "winter", "test")

    assert data.seasonal is None
    assert data.watchlist is None


@pytest.mark.asyncio
async def test_databuilder_without_season():
    b = (
        recommender_builder.RecommenderBuilder()
        .provider(test_provider.AsyncProviderStub())
        .engine("fake")
    )

    actual = b.build()
    data = await actual.get_dataset(2023, None, "test")

    assert isinstance(actual, recommender.AnimeRecommender)
    assert actual.provider is not None
    assert actual.databuilder is not None
    assert actual.engine is not None

    assert "Copper Kamuy 4th Season" in data.seasonal["title"].to_list()


def test_builder_creation_returns_correct_builders():
    assert (
        recommender_builder.create_builder("anilist")._provider.__class__
        == providers.anilist.AniListProvider
    )
    assert (
        recommender_builder.create_builder("myanimelist")._provider.__class__
        == providers.myanimelist.MyAnimeListProvider
    )

    assert (
        recommender_builder.create_builder("faulty-provider")._provider.__class__
        == providers.mixed.MixedProvider
    )


def test_builder_passes_with_or_without_cache(mocker):
    assert (
        recommender_builder.create_builder("anilist")._provider.__class__
        == providers.anilist.AniListProvider
    )

    mocker.patch("animeippo.cache.RedisCache", CacheStub)

    assert (
        recommender_builder.create_builder("anilist")._provider.__class__
        == providers.anilist.AniListProvider
    )
