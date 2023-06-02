from animeippo.providers import myanimelist
from tests import test_data

import pytest


class SessionStub:
    dictionary = {}

    def __init__(self, dictionary):
        self.dictionary = dictionary

    def get(self, key, *args, **kwargs):
        return self.dictionary.get(key)

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def __aenter__(self):
        return self


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
async def test_mal_user_anime_list_can_be_fetched(mocker):
    provider = myanimelist.MyAnimeListProvider()

    user = "Janiskeisari"

    response = ResponseStub(test_data.MAL_USER_LIST)
    mocker.patch("aiohttp.ClientSession.get", return_value=response)

    animelist = await provider.get_user_anime_list(user)

    assert "Hellsingfårs" in animelist["title"].values


@pytest.mark.asyncio
async def test_mal_seasonal_anime_list_can_be_fetched(mocker):
    provider = myanimelist.MyAnimeListProvider()

    year = "2023"
    season = "winter"

    response = ResponseStub(test_data.MAL_SEASONAL_LIST)
    mocker.patch("aiohttp.ClientSession.get", return_value=response)

    animelist = await provider.get_seasonal_anime_list(year, season)

    assert "Shingeki no Kyojin: The Fake Season" in animelist["title"].values


@pytest.mark.asyncio
async def test_mal_related_anime_can_be_fetched(mocker):
    provider = myanimelist.MyAnimeListProvider()

    anime_id = 30

    response = ResponseStub(test_data.MAL_RELATED_ANIME)

    mocker.patch("aiohttp.ClientSession.get", return_value=response)

    details = await provider.get_related_anime(anime_id)

    assert details.index.tolist() == [31]


@pytest.mark.asyncio
async def test_mal_related_anime_does_not_fail_with_invalid_data(mocker):
    provider = myanimelist.MyAnimeListProvider()

    anime_id = 30

    response = ResponseStub({"related_anime": []})
    mocker.patch("aiohttp.ClientSession.get", return_value=response)

    details = await provider.get_related_anime(anime_id)

    assert details.index.tolist() == []


@pytest.mark.asyncio
async def test_get_next_page_returns_succesfully(mocker):
    response1 = ResponseStub({"data": [{"test": "test"}], "paging": {"next": "page2"}})
    response2 = ResponseStub({"data": [{"test2": "test2"}], "paging": {"next": "page3"}})
    response3 = ResponseStub({"data": [{"test3": "test3"}]})

    pages = [response1, response2, response3]

    mocker.patch("aiohttp.ClientSession.get", side_effect=pages)

    session = SessionStub({"page1": response1, "page2": response2, "page3": response3})
    final_pages = [
        await myanimelist.MyAnimeListConnection().requests_get_next_page(session, await page.json())
        for page in pages
    ]

    assert len(final_pages) == 3
    assert final_pages[0] == await response2.json()
    assert final_pages[1] == await response3.json()
    assert final_pages[2] is None


@pytest.mark.asyncio
async def test_get_all_pages_returns_all_pages(mocker):
    response1 = ResponseStub({"data": [{"test": "test"}], "paging": {"next": "page2"}})
    response2 = ResponseStub({"data": [{"test2": "test2"}], "paging": {"next": "page3"}})
    response3 = ResponseStub({"data": [{"test3": "test3"}]})

    mocker.patch("animeippo.providers.myanimelist.MAL_API_URL", "FAKE")
    first_page_url = "/users/kamina69/animelist"

    response = ResponseStub({"related_anime": []})
    mocker.patch("aiohttp.ClientSession.get", return_value=response)

    mock_session = SessionStub(
        {"FAKE" + first_page_url: response1, "page2": response2, "page3": response3}
    )

    final_pages = list(
        [
            page
            async for page in myanimelist.MyAnimeListConnection().requests_get_all_pages(
                mock_session, first_page_url, None
            )
        ]
    )

    assert len(final_pages) == 3
    assert final_pages[0] == await response1.json()


@pytest.mark.asyncio
async def test_request_page_succesfully_exists_with_blank_page():
    page = None
    mock_session = SessionStub({})

    actual = await myanimelist.MyAnimeListConnection().requests_get_next_page(mock_session, page)

    assert actual is None


@pytest.mark.asyncio
async def test_reqest_does_not_fail_catastrophically_when_response_is_empty(mocker):
    response1 = ResponseStub(dict())

    mocker.patch("animeippo.providers.myanimelist.MAL_API_URL", "FAKE")
    first_page_url = "/users/kamina69/animelist"

    mock_session = SessionStub({"FAKE" + first_page_url: response1})

    pages = list(
        [
            page
            async for page in myanimelist.MyAnimeListConnection().requests_get_all_pages(
                mock_session, first_page_url, None
            )
        ]
    )

    assert len(pages) == 0


def test_features_can_be_fetched():
    provider = myanimelist.MyAnimeListProvider()

    features = provider.get_features()

    assert len(features) > 0
    assert "genres" in features


@pytest.mark.asyncio
async def test_anilist_returns_None_with_empty_parameters():
    provider = myanimelist.MyAnimeListProvider()

    related_anime = await provider.get_related_anime(None)
    seasonal_anime = await provider.get_seasonal_anime_list(None, None)
    user_anime = await provider.get_user_anime_list(None)

    assert related_anime is None
    assert seasonal_anime is None
    assert user_anime is None
