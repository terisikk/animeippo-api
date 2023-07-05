from animeippo.recommendation import builder, recommender
import animeippo.providers as providers
import pandas as pd

from tests import test_data

import pytest


class AsyncProviderStub:
    def __init__(
        self,
        seasonal=test_data.FORMATTED_MAL_SEASONAL_LIST,
        user=test_data.FORMATTED_MAL_USER_LIST,
        cache=None,
    ):
        self.seasonal = seasonal
        self.user = user
        self.cache = cache

    async def get_seasonal_anime_list(self, *args, **kwargs):
        return pd.DataFrame(self.seasonal).set_index("id")

    async def get_user_anime_list(self, *args, **kwargs):
        return pd.DataFrame(self.user).set_index("id")

    async def get_related_anime(self, *args, **kwargs):
        return pd.DataFrame()

    def get_features(self, *args, **kwargs):
        return ["genres"]


class FaultyProviderStub:
    def __init__(
        self,
        cache=None,
    ):
        self.cache = cache

    async def get_seasonal_anime_list(self, *args, **kwargs):
        return None

    async def get_user_anime_list(self, *args, **kwargs):
        return None

    async def get_related_anime(self, *args, **kwargs):
        return None

    def get_features(self, *args, **kwargs):
        return ["genres"]


@pytest.mark.asyncio
async def test_Recommenderbuilder_with_anilist():
    b = (
        builder.RecommenderBuilder()
        .provider(AsyncProviderStub())
        .model("fake")
        .databuilder(builder.construct_anilist_data)
    )

    actual = b.build()
    data = await actual.get_dataset("2023", "winter", "test")

    assert isinstance(actual, recommender.AnimeRecommender)
    assert actual.provider is not None
    assert actual.databuilder is not None
    assert actual.engine is not None

    assert "Copper Kamuy 4th Season" in data.seasonal["title"].to_list()


@pytest.mark.asyncio
async def test_Recommenderbuilder_with_mal():
    b = (
        builder.RecommenderBuilder()
        .provider(AsyncProviderStub())
        .model("fake")
        .databuilder(builder.construct_myanimelist_data)
    )

    actual = b.build()
    data = await actual.get_dataset("2023", "winter", "test")

    assert isinstance(actual, recommender.AnimeRecommender)
    assert actual.provider is not None
    assert actual.databuilder is not None
    assert actual.engine is not None

    assert "Copper Kamuy 4th Season" in data.seasonal["title"].to_list()


@pytest.mark.asyncio
async def test_mal_databuilder_does_not_fail_with_missing_data():
    b = (
        builder.RecommenderBuilder()
        .provider(FaultyProviderStub())
        .model("fake")
        .databuilder(builder.construct_myanimelist_data)
    )

    actual = b.build()
    data = await actual.get_dataset("2023", "winter", "test")

    assert data.seasonal is None
    assert data.watchlist is None
    assert data.feature_names == ["genres"]


@pytest.mark.asyncio
async def test_anilist_databuilder_does_not_fail_with_missing_data():
    b = (
        builder.RecommenderBuilder()
        .provider(FaultyProviderStub())
        .model("fake")
        .databuilder(builder.construct_anilist_data)
    )

    actual = b.build()
    data = await actual.get_dataset("2023", "winter", "test")

    assert data.seasonal is None
    assert data.watchlist is None
    assert data.feature_names == ["genres"]


@pytest.mark.asyncio
async def test_databuilder_without_season():
    b = (
        builder.RecommenderBuilder()
        .provider(AsyncProviderStub())
        .model("fake")
        .databuilder(builder.construct_anilist_data)
    )

    actual = b.build()
    data = await actual.get_dataset("2023", None, "test")

    assert isinstance(actual, recommender.AnimeRecommender)
    assert actual.provider is not None
    assert actual.databuilder is not None
    assert actual.engine is not None

    assert "Copper Kamuy 4th Season" in data.seasonal["title"].to_list()


def test_builder_creation_returns_correct_builders():
    assert (
        builder.create_builder("anilist")._provider.__class__ == providers.anilist.AniListProvider
    )
    assert (
        builder.create_builder("myanimelist")._provider.__class__
        == providers.myanimelist.MyAnimeListProvider
    )

    assert (
        builder.create_builder("faulty-provider")._provider.__class__
        == providers.myanimelist.MyAnimeListProvider
    )


def test_status_data_is_filled_to_dataset():
    watchlist = pd.DataFrame(
        {"id": [110, 120, 130], "user_status": ["completed", "watching", "completed"]}
    ).set_index("id")

    seasonal = pd.DataFrame(
        {"id": [110, 120, 140], "title": ["Test 1", "Test 2", "Test 3"]}
    ).set_index("id")

    seasonal = builder.fill_user_status_data_from_watchlist(seasonal, watchlist)

    assert "user_status" in seasonal.columns
    assert seasonal.loc[110, "user_status"] == "completed"
    assert seasonal.loc[120, "user_status"] == "watching"
    assert pd.isnull(seasonal.loc[140, "user_status"])
    assert len(seasonal) == 3
