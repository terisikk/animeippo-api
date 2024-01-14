from animeippo.providers import myanimelist
from tests import test_data

import pytest

from tests.providers.myanimelist.stubs import ResponseStub


@pytest.mark.asyncio
async def test_mal_user_anime_list_can_be_fetched(mocker):
    provider = myanimelist.MyAnimeListProvider()

    user = "Janiskeisari"

    response = ResponseStub(test_data.MAL_USER_LIST)
    mocker.patch("aiohttp.ClientSession.get", return_value=response)

    animelist = await provider.get_user_anime_list(user)

    assert "Hellsingfårs" in animelist["title"].to_list()


@pytest.mark.asyncio
async def test_mal_seasonal_anime_list_can_be_fetched(mocker):
    provider = myanimelist.MyAnimeListProvider()

    year = "2023"
    season = "winter"

    response = ResponseStub(test_data.MAL_SEASONAL_LIST)
    mocker.patch("aiohttp.ClientSession.get", return_value=response)

    animelist = await provider.get_seasonal_anime_list(year, season)

    assert "Shingeki no Kyojin: The Fake Season" in animelist["title"].to_list()


@pytest.mark.asyncio
async def test_mal_fetches_several_anime_list_when_season_is_none(mocker):
    provider = myanimelist.MyAnimeListProvider()

    year = "2023"
    season = None

    response = ResponseStub(test_data.MAL_SEASONAL_LIST)
    mocker.patch("aiohttp.ClientSession.get", return_value=response)

    animelist = await provider.get_seasonal_anime_list(year, season)

    assert "Shingeki no Kyojin: The Fake Season" in animelist["title"].to_list()


@pytest.mark.asyncio
async def test_mal_user_manga_list_can_be_fetched(mocker):
    provider = myanimelist.MyAnimeListProvider()

    user = "Janiskeisari"

    response = ResponseStub(test_data.MAL_MANGA_LIST)
    mocker.patch("aiohttp.ClientSession.get", return_value=response)

    animelist = await provider.get_user_manga_list(user)

    assert "Daadaa dandaddaa" in animelist["title"].to_list()


@pytest.mark.asyncio
async def test_mal_related_anime_can_be_fetched(mocker):
    provider = myanimelist.MyAnimeListProvider()

    anime_id = 30

    response = ResponseStub(test_data.MAL_RELATED_ANIME)

    mocker.patch("aiohttp.ClientSession.get", return_value=response)

    details = await provider.get_related_anime(anime_id)

    assert details == [31]


@pytest.mark.asyncio
async def test_mal_related_anime_does_not_fail_with_invalid_data(mocker):
    provider = myanimelist.MyAnimeListProvider()

    anime_id = 30

    response = ResponseStub({"related_anime": []})
    mocker.patch("aiohttp.ClientSession.get", return_value=response)

    details = await provider.get_related_anime(anime_id)

    assert details == []


def test_features_can_be_fetched():
    provider = myanimelist.MyAnimeListProvider()

    features = provider.get_feature_fields()

    assert len(features) > 0
    assert "genres" in features


@pytest.mark.asyncio
async def test_mal_returns_None_with_empty_parameters():
    provider = myanimelist.MyAnimeListProvider()

    related_anime = await provider.get_related_anime(None)
    seasonal_anime = await provider.get_seasonal_anime_list(None, None)
    user_anime = await provider.get_user_anime_list(None)
    user_manga = await provider.get_user_manga_list(None)

    assert related_anime is None
    assert seasonal_anime is None
    assert user_anime is None
    assert user_manga is None
