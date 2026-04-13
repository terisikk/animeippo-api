import aiohttp
import pytest

import animeippo.providers.anilist.connection
from animeippo.providers import anilist
from tests import test_data


class ResponseStub:
    def __init__(self, dictionary):
        self.dictionary = dictionary
        self.status = 200
        self.headers = {"X-RateLimit-Remaining": "90", "X-RateLimit-Limit": "90"}

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


class SessionStub:
    """Stub that simulates aiohttp.ClientSession for request_single tests."""

    def __init__(self, response):
        self.response = response

    def post(self, *args, **kwargs):
        return self.response


@pytest.mark.asyncio
async def test_ani_user_anime_list_can_be_fetched(mocker):
    provider = anilist.AniListProvider()

    user = "Janiskeisari"

    response = ResponseStub(test_data.ANI_USER_LIST)
    mocker.patch("aiohttp.ClientSession.post", return_value=response)

    animelist = await provider.get_user_anime_list(user)

    assert "Dr. STRONK: OLD WORLD" in animelist["title"]


@pytest.mark.asyncio
async def test_ani_seasonal_anime_list_can_be_fetched(mocker):
    provider = anilist.AniListProvider()

    year = "2023"
    season = "winter"

    response = ResponseStub(test_data.ANI_SEASONAL_LIST)
    mocker.patch("aiohttp.ClientSession.post", return_value=response)

    animelist = await provider.get_seasonal_anime_list(year, season)

    assert "EDENS KNOCK-OFF 2nd Season" in animelist["title"]


@pytest.mark.asyncio
async def test_all_yearly_season_can_be_fetched_when_season_is_none(mocker):
    provider = anilist.AniListProvider()

    year = "2023"
    season = None

    response = ResponseStub(test_data.ANI_SEASONAL_LIST)
    mocker.patch("aiohttp.ClientSession.post", return_value=response)

    animelist = await provider.get_seasonal_anime_list(year, season)

    assert "EDENS KNOCK-OFF 2nd Season" in animelist["title"]


@pytest.mark.asyncio
async def test_ani_user_manga_list_can_be_fetched(mocker):
    provider = anilist.AniListProvider()

    response = ResponseStub(test_data.ANI_MANGA_LIST)
    mocker.patch("aiohttp.ClientSession.post", return_value=response)

    animelist = await provider.get_user_manga_list("Janiskeisari")

    assert "Dr. BONK: BONK BATTLES" in animelist["title"]


def test_ani_related_anime_returns_none():
    provider = anilist.AniListProvider()

    animelist = provider.get_related_anime(0)

    assert animelist is None


@pytest.mark.asyncio
async def test_get_single_returns_succesfully(mocker):
    response_json = {"data": [{"test": "test"}], "pageInfo": {"hasNextPage": False}}

    response = ResponseStub(response_json)
    session = SessionStub(response)

    page = await animeippo.providers.anilist.AnilistConnection().request_single(session, "test", {})

    assert page == await response.json()


@pytest.mark.asyncio
async def test_get_all_pages_returns_all_pages(mocker):
    response1 = {
        "data": {
            "Page": {
                "media": {"test": "test2"},
                "pageInfo": {"hasNextPage": True, "currentPage": 1, "lastPage": 3},
            }
        }
    }
    response2 = {
        "data": {
            "Page": {
                "media": {"test": "test2"},
                "pageInfo": {"hasNextPage": True, "currentPage": 2, "lastPage": 3},
            }
        }
    }
    response3 = {
        "data": {
            "Page": {
                "media": {"test": "test1"},
                "pageInfo": {"hasNextPage": False, "currentPage": 3, "lastPage": 3},
            }
        }
    }

    mocker.patch(
        "aiohttp.ClientSession.post",
        side_effect=[ResponseStub(response1), ResponseStub(response2), ResponseStub(response3)],
    )

    session = aiohttp.ClientSession()
    final_pages = [
        page
        async for page in animeippo.providers.anilist.AnilistConnection().get_all_pages(
            session, "", {}
        )
    ]
    await session.close()

    assert len(final_pages) == 3
    assert final_pages[0] == response1["data"]["Page"]
    assert final_pages[2] == response3["data"]["Page"]


@pytest.mark.asyncio
async def test_request_does_not_fail_catastrophically_when_response_is_empty(mocker):
    response = ResponseStub({})
    mocker.patch("aiohttp.ClientSession.post", return_value=response)

    session = aiohttp.ClientSession()
    all_pages = [
        page
        async for page in animeippo.providers.anilist.AnilistConnection().get_all_pages(
            session, "", {}
        )
    ]
    await session.close()

    assert len(all_pages) == 1
    assert all_pages[0] is None


@pytest.mark.asyncio
async def test_get_all_pages_returns_single_page_when_no_next_page(mocker):
    response = {
        "data": {
            "Page": {
                "media": {"test": "data"},
                "pageInfo": {"hasNextPage": False, "currentPage": 1, "lastPage": 1},
            }
        }
    }

    mocker.patch("aiohttp.ClientSession.post", return_value=ResponseStub(response))

    session = aiohttp.ClientSession()
    final_pages = [
        page
        async for page in animeippo.providers.anilist.AnilistConnection().get_all_pages(
            session, "", {}
        )
    ]
    await session.close()

    assert len(final_pages) == 1
    assert final_pages[0] == response["data"]["Page"]


@pytest.mark.asyncio
async def test_get_all_pages_with_null_page_data_in_response(mocker):
    response1 = {
        "data": {
            "Page": {
                "media": {"test": "test1"},
                "pageInfo": {"hasNextPage": True, "currentPage": 1, "lastPage": 2},
            }
        }
    }
    response2 = {
        "data": {
            "Page": None  # Null page data in second response
        }
    }

    mocker.patch(
        "aiohttp.ClientSession.post",
        side_effect=[ResponseStub(response1), ResponseStub(response2)],
    )

    session = aiohttp.ClientSession()
    final_pages = [
        page
        async for page in animeippo.providers.anilist.AnilistConnection().get_all_pages(
            session, "", {}
        )
    ]
    await session.close()

    # Should only get the first page, null page should be filtered out
    assert len(final_pages) == 1
    assert final_pages[0] == response1["data"]["Page"]


@pytest.mark.asyncio
async def test_rate_limit_warning_logged_when_remaining_low(mocker):
    response_stub = ResponseStub({"data": "test"})
    response_stub.headers = {"X-RateLimit-Remaining": "5", "X-RateLimit-Limit": "90"}

    session = SessionStub(response_stub)

    connection = animeippo.providers.anilist.AnilistConnection()
    await connection.request_single(session, "", {})

    assert connection.rate_remaining == 5
    assert connection.rate_limit == 90


@pytest.mark.asyncio
async def test_rate_limit_retries_on_429(mocker):
    rate_limited_stub = ResponseStub({"data": None})
    rate_limited_stub.status = 429
    rate_limited_stub.headers = {
        "X-RateLimit-Remaining": "0",
        "X-RateLimit-Limit": "90",
        "Retry-After": "0",
    }

    ok_stub = ResponseStub({"data": "success"})

    mocker.patch("asyncio.sleep", return_value=None)

    call_count = 0

    class RetrySessionStub:
        def post(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return rate_limited_stub
            return ok_stub

    connection = animeippo.providers.anilist.AnilistConnection()
    result = await connection.request_single(RetrySessionStub(), "", {})

    assert result == {"data": "success"}


@pytest.mark.asyncio
async def test_error_status_raises_client_error(mocker):
    error_stub = ResponseStub({"errors": [{"message": "Internal Server Error"}]})
    error_stub.status = 500

    session = SessionStub(error_stub)

    connection = animeippo.providers.anilist.AnilistConnection()

    with pytest.raises(aiohttp.ClientResponseError):
        await connection.request_single(session, "", {})


@pytest.mark.asyncio
async def test_not_found_status_raises_client_error():
    error_stub = ResponseStub({"errors": [{"message": "User not found"}]})
    error_stub.status = 404

    session = SessionStub(error_stub)

    connection = animeippo.providers.anilist.AnilistConnection()

    with pytest.raises(aiohttp.ClientResponseError):
        await connection.request_single(session, "", {})


@pytest.mark.asyncio
async def test_anilist_returns_None_with_empty_parameters():
    provider = anilist.AniListProvider()

    seasonal_anime = await provider.get_seasonal_anime_list(None, None)
    user_anime = await provider.get_user_anime_list(None)
    user_manga = await provider.get_user_manga_list(None)

    assert seasonal_anime is None
    assert user_anime is None
    assert user_manga is None


def test_anilist_nsfw_tags_function_returns_nsfw_tags():
    provider = anilist.AniListProvider()

    tags = provider.get_nsfw_tags()

    assert tags is not None
    assert "Bondage" in tags  # Bondage is an NSFW tag
    # Mystery is not an NSFW tag
    assert "Mystery" not in tags


def test_anilist_get_genres_returns_genres():
    provider = anilist.AniListProvider()

    genres = provider.get_genres()

    assert genres is not None
    assert "Action" in genres


@pytest.mark.asyncio
async def test_custom_lists_are_filtered_out_from_user_anime_list(mocker):
    provider = anilist.AniListProvider()

    user = "Janiskeisari"

    response = ResponseStub(test_data.ANI_USER_LIST)
    mocker.patch("aiohttp.ClientSession.post", return_value=response)

    animelist = await provider.get_user_anime_list(user)

    assert "Dr. STRONK: OLD WORLD" in animelist["title"]
    # Ensure custom lists are filtered out
    assert "Custom List Anime" not in animelist["title"]


@pytest.mark.asyncio
async def test_custom_lists_are_filtered_out_from_user_manga_list(mocker):
    provider = anilist.AniListProvider()

    user = "Janiskeisari"

    response = ResponseStub(test_data.ANI_USER_LIST)
    mocker.patch("aiohttp.ClientSession.post", return_value=response)

    animelist = await provider.get_user_manga_list(user)

    assert "Dr. STRONK: OLD WORLD" in animelist["title"]
    # Ensure custom lists are filtered out
    assert "Custom List Anime" not in animelist["title"]
