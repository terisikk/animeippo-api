from animeippo.recommendation import builder, recommender
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


def test_new_builder_can_be_instantiated():
    class ConcreteRecommenderBuilder(builder.AbstractRecommenderBuilder):
        def __init__(self):
            super().__init__()

        def build(self):
            super().build()
            provider = self._build_provider()
            self._build_databuilder(provider, None, None, None)
            self._build_model()

        def _build_provider(self):
            return super()._build_provider()

        def _build_databuilder(self, provider, user, year, season):
            return super()._build_databuilder(provider, user, year, season)

        def _build_model(self):
            return super()._build_model()

    actual = ConcreteRecommenderBuilder()
    actual.build()

    assert issubclass(actual.__class__, builder.AbstractRecommenderBuilder)


@pytest.mark.asyncio
async def test_AniListRecommenderbuilder(mocker):
    mocker.patch("animeippo.providers.anilist.AniListProvider", AsyncProviderStub)

    b = builder.AniListRecommenderBuilder()

    actual = b.build()
    data = await actual.databuilder("2023", "winter", "test")

    assert isinstance(actual, recommender.AnimeRecommender)
    assert actual.provider is not None
    assert actual.databuilder is not None
    assert actual.engine is not None

    assert "Golden Kamuy 4th Season" in data.seasonal["title"].to_list()


@pytest.mark.asyncio
async def test_MyAnimeListRecommenderbuilder(mocker):
    mocker.patch("animeippo.providers.myanimelist.MyAnimeListProvider", AsyncProviderStub)

    b = builder.MyAnimeListRecommenderBuilder()

    actual = b.build()
    data = await actual.databuilder("2023", "winter", "test")

    assert isinstance(actual, recommender.AnimeRecommender)
    assert actual.provider is not None
    assert actual.databuilder is not None
    assert actual.engine is not None

    assert "Golden Kamuy 4th Season" in data.seasonal["title"].to_list()


@pytest.mark.asyncio
async def test_mal_databuilder_does_not_fail_with_missing_data(mocker):
    mocker.patch("animeippo.providers.myanimelist.MyAnimeListProvider", FaultyProviderStub)

    b = builder.MyAnimeListRecommenderBuilder()

    actual = b.build()
    data = await actual.databuilder("2023", "winter", "test")

    assert data.seasonal is None
    assert data.watchlist is None
    assert data.features == ["genres"]


@pytest.mark.asyncio
async def test_anilist_databuilder_does_not_fail_with_missing_data(mocker):
    mocker.patch("animeippo.providers.anilist.AniListProvider", FaultyProviderStub)

    b = builder.AniListRecommenderBuilder()

    actual = b.build()
    data = await actual.databuilder("2023", "winter", "test")

    assert data.seasonal is None
    assert data.watchlist is None
    assert data.features == ["genres"]


def test_builder_creation_returns_correct_builders():
    assert isinstance(builder.create_builder("anilist"), builder.AniListRecommenderBuilder)
    assert isinstance(builder.create_builder("myanimelist"), builder.MyAnimeListRecommenderBuilder)

    assert isinstance(
        builder.create_builder("faulty_defaults_to_mal"), builder.MyAnimeListRecommenderBuilder
    )
