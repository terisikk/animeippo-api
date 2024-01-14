import pytest
import polars as pl
import animeippo.providers.provider_builder

from animeippo.recommendation import recommender, recommender_builder
import animeippo.providers as providers

from tests import test_provider


class CacheStub:
    def is_available(self):
        return self.is_available


@pytest.mark.asyncio
async def test_Recommenderbuilder_with_anilist():
    b = (
        recommender_builder.RecommenderBuilder()
        .provider(test_provider.AsyncProviderStub())
        .model("fake")
        .databuilder(animeippo.providers.provider_builder.construct_anilist_data)
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
        .model("fake")
        .databuilder(animeippo.providers.provider_builder.construct_myanimelist_data)
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
        .model("fake")
        .databuilder(animeippo.providers.provider_builder.construct_myanimelist_data)
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
        .model("fake")
        .databuilder(animeippo.providers.provider_builder.construct_anilist_data)
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
        .model("fake")
        .databuilder(animeippo.providers.provider_builder.construct_anilist_data)
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


def test_status_data_is_filled_to_dataset():
    watchlist = pl.DataFrame(
        {
            "id": [110, 120, 130],
            "user_status": [
                "completed",
                "watching",
                "completed",
            ],
        }
    )

    seasonal = pl.DataFrame(
        {
            "id": [110, 120, 140],
            "title": [
                "Test 1",
                "Test 2",
                "Test 3",
            ],
        }
    )

    seasonal = animeippo.providers.provider_builder.fill_user_status_data_from_watchlist(
        seasonal, watchlist
    )

    assert "user_status" in seasonal.columns
    assert seasonal.item(0, "user_status") == "completed"
    assert seasonal.item(1, "user_status") == "watching"
    assert seasonal.item(2, "user_status") is None
    assert len(seasonal) == 3
