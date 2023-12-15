import pytest

from animeippo.providers import mixed_provider

from tests import test_data


class ResponseStub:
    dictionary = {}

    def __init__(self, dictionary):
        self.dictionary = dictionary

    async def get(self, key):
        return self.dictionary.get(key)

    async def json(self):
        return self.dictionary

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def __aenter__(self):
        return self

    def raise_for_status(self):
        pass


@pytest.mark.asyncio
async def test_mixed_provider_user_anime_can_be_fetched(mocker):
    provider = mixed_provider.MixedProvider()

    mocker.patch.object(
        provider.ani_provider.connection,
        "request_paginated",
        return_value=test_data.MIXED_USER_LIST_ANI,
    )
    mocker.patch.object(
        provider.mal_provider.connection,
        "request_anime_list",
        return_value=test_data.MIXED_USER_LIST_MAL,
    )

    user_anime = await provider.get_user_anime_list(1)

    assert len(user_anime) == 2
    assert user_anime.iloc[0]["title"] == "Neon Genesis Evangelion"
    assert user_anime.iloc[1]["title"] == "Hellsingfårs"


@pytest.mark.asyncio
async def test_mixed_provider_returns_None_with_empty_parameters():
    provider = mixed_provider.MixedProvider()

    seasonal_anime = await provider.get_seasonal_anime_list(None, None)
    user_anime = await provider.get_user_anime_list(None)
    user_manga = await provider.get_user_manga_list(None)

    assert seasonal_anime is None
    assert user_anime is None
    assert user_manga is None


@pytest.mark.asyncio
async def test_mixed_provider_seasonal_anime_list_can_be_fetched(mocker):
    provider = mixed_provider.MixedProvider()

    year = "2023"
    season = "winter"

    mocker.patch.object(
        provider.ani_provider.connection,
        "request_paginated",
        return_value=test_data.MIXED_ANI_SEASONAL_LIST,
    )

    animelist = await provider.get_seasonal_anime_list(year, season)

    assert "EDENS KNOCK-OFF 2nd Season" in animelist["title"].values


@pytest.mark.asyncio
async def test_mixed_provider_yearly_list_can_be_fetched_when_season_is_none(mocker):
    provider = mixed_provider.MixedProvider()

    year = "2023"
    season = None

    mocker.patch.object(
        provider.ani_provider.connection,
        "request_paginated",
        return_value=test_data.MIXED_ANI_SEASONAL_LIST,
    )

    animelist = await provider.get_seasonal_anime_list(year, season)

    assert "EDENS KNOCK-OFF 2nd Season" in animelist["title"].values


def test_mixed_provider_related_anime_returns_none():
    provider = mixed_provider.MixedProvider()

    animelist = provider.get_related_anime(0)

    assert animelist is None
