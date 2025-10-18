import pytest

import animeippo.providers.anilist.connection
from animeippo.providers import anilist
from tests import test_data


class ResponseStub:
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
    mocker.patch("aiohttp.ClientSession.post", return_value=response)

    page = await animeippo.providers.anilist.AnilistConnection().request_single("test", {})

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

    final_pages = [
        page async for page in animeippo.providers.anilist.AnilistConnection().get_all_pages("", {})
    ]

    assert len(final_pages) == 3
    assert final_pages[0] == response1["data"]["Page"]
    assert final_pages[2] == response3["data"]["Page"]


@pytest.mark.asyncio
async def test_request_does_not_fail_catastrophically_when_response_is_empty(mocker):
    response = {}

    response = ResponseStub({})
    mocker.patch("aiohttp.ClientSession.post", return_value=response)

    all_pages = [
        page async for page in animeippo.providers.anilist.AnilistConnection().get_all_pages("", {})
    ]

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

    final_pages = [
        page async for page in animeippo.providers.anilist.AnilistConnection().get_all_pages("", {})
    ]

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

    final_pages = [
        page async for page in animeippo.providers.anilist.AnilistConnection().get_all_pages("", {})
    ]

    # Should only get the first page, null page should be filtered out
    assert len(final_pages) == 1
    assert final_pages[0] == response1["data"]["Page"]


def test_features_can_be_fetched():
    provider = anilist.AniListProvider()

    features = provider.get_feature_fields()

    assert len(features) > 0
    assert "genres" in features


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
    assert 246 in tags  # Bondage (ID 246 is an NSFW tag)
    # Mystery is not an NSFW tag, but we need to check by ID or ensure no genre IDs are in NSFW tags
    # Since Mystery is a genre (not a tag with an ID in our system), we just verify it's not there
    assert "Mystery" not in tags  # Genres are strings, not IDs


def test_anilist_get_genres_returns_genres():
    provider = anilist.AniListProvider()

    genres = provider.get_genres()

    assert genres is not None
    assert "Action" in genres


def test_anilist_get_genres_from_cache(mocker):
    """Test that genres are fetched from cache when available."""
    mock_cache = mocker.Mock()
    mock_cache.is_available.return_value = True
    mock_cache.get_json.return_value = ["Action", "Comedy", "Drama"]

    provider = anilist.AniListProvider(cache=mock_cache)
    genres = provider.get_genres()

    mock_cache.get_json.assert_called_once_with("anilist:genres")
    assert genres == {"Action", "Comedy", "Drama"}


def test_anilist_get_genres_fallback_to_static_when_cache_empty(mocker):
    """Test that genres fallback to static data when cache returns None."""
    mock_cache = mocker.Mock()
    mock_cache.is_available.return_value = True
    mock_cache.get_json.return_value = None

    provider = anilist.AniListProvider(cache=mock_cache)
    genres = provider.get_genres()

    mock_cache.get_json.assert_called_once_with("anilist:genres")
    assert "Action" in genres  # From static data


def test_anilist_get_nsfw_tags_from_cache(mocker):
    """Test that NSFW tags are fetched from cache when available."""
    mock_cache = mocker.Mock()
    mock_cache.is_available.return_value = True
    # Note: The pre-loading script stores tag names in cache for NSFW tags
    # But the code converts them to a set, so they remain as names
    mock_cache.get_json.return_value = ["Bondage", "Nudity"]

    provider = anilist.AniListProvider(cache=mock_cache)
    tags = provider.get_nsfw_tags()

    mock_cache.get_json.assert_called_once_with("anilist:nsfw_tags")
    assert tags == {"Bondage", "Nudity"}


def test_anilist_get_nsfw_tags_fallback_to_static_when_cache_empty(mocker):
    """Test that NSFW tags fallback to static data when cache returns None."""
    mock_cache = mocker.Mock()
    mock_cache.is_available.return_value = True
    mock_cache.get_json.return_value = None

    provider = anilist.AniListProvider(cache=mock_cache)
    tags = provider.get_nsfw_tags()

    mock_cache.get_json.assert_called_once_with("anilist:nsfw_tags")
    assert 246 in tags  # Bondage (ID 246) from static data


@pytest.mark.asyncio
async def test_get_session_creates_session():
    connection = animeippo.providers.anilist.AnilistConnection()

    session = await connection.get_session()

    assert session is not None
    assert not session.closed

    await connection.close()


@pytest.mark.asyncio
async def test_get_session_reuses_existing_session():
    connection = animeippo.providers.anilist.AnilistConnection()

    session1 = await connection.get_session()
    session2 = await connection.get_session()

    assert session1 is session2

    await connection.close()


@pytest.mark.asyncio
async def test_close_closes_session():
    connection = animeippo.providers.anilist.AnilistConnection()

    session = await connection.get_session()
    assert not session.closed

    await connection.close()

    assert session.closed
    assert connection._session is None


@pytest.mark.asyncio
async def test_close_does_nothing_when_no_session():
    connection = animeippo.providers.anilist.AnilistConnection()

    # Should not raise an error
    await connection.close()

    assert connection._session is None


@pytest.mark.asyncio
async def test_context_manager_closes_session():
    """Test that using AnilistConnection as async context manager properly closes session."""
    async with animeippo.providers.anilist.AnilistConnection() as connection:
        session = await connection.get_session()
        assert not session.closed

    # Session should be closed after exiting context
    assert session.closed


@pytest.mark.asyncio
async def test_context_manager_with_exception():
    """Test that session is closed even if exception occurs in context."""
    connection = None
    session = None

    try:
        async with animeippo.providers.anilist.AnilistConnection() as conn:
            connection = conn
            session = await connection.get_session()
            assert not session.closed
            raise ValueError("test error")
    except ValueError:
        pass

    # Session should still be closed after exception
    assert session.closed


@pytest.mark.asyncio
async def test_anilist_provider_context_manager():
    """Test that AniListProvider properly manages connection lifecycle."""
    async with anilist.AniListProvider() as provider:
        # Provider should work normally
        assert provider.connection is not None

    # Connection should be closed after context exit
    # We can't directly check the session, but we can verify close was called
    assert provider.connection._session is None or provider.connection._session.closed


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
