import pytest

from animeippo.providers.myanimelist.connection import HTTP_UNAUTHORIZED, MyAnimeListConnection

from .stubs import ResponseStub, SessionStub


@pytest.mark.asyncio
async def test_get_next_page_returns_succesfully(mocker):
    mocker.patch("asyncio.sleep", return_value=None)

    response2 = ResponseStub({"data": [{"test2": "test2"}], "paging": {"next": "page3"}})
    response3 = ResponseStub({"data": [{"test3": "test3"}]})

    page_with_next = {"data": [{"test": "test"}], "paging": {"next": "page2"}}
    page_without_next = {"data": [{"test3": "test3"}]}

    session = SessionStub({"page2": response2, "page3": response3})
    conn = MyAnimeListConnection()

    result = await conn.requests_get_next_page(session, page_with_next)
    assert result == await response2.json()

    result = await conn.requests_get_next_page(session, page_without_next)
    assert result is None


@pytest.mark.asyncio
async def test_get_all_pages_returns_all_pages(mocker):
    response1 = ResponseStub({"data": [{"test": "test"}], "paging": {"next": "page2"}})
    response2 = ResponseStub({"data": [{"test2": "test2"}], "paging": {"next": "page3"}})
    response3 = ResponseStub({"data": [{"test3": "test3"}]})

    mocker.patch("animeippo.providers.myanimelist.connection.MAL_API_URL", "FAKE")
    mocker.patch("asyncio.sleep", return_value=None)

    mock_session = SessionStub(
        {"FAKE/users/test/animelist": response1, "page2": response2, "page3": response3}
    )

    conn = MyAnimeListConnection()
    final_pages = [
        page
        async for page in conn.requests_get_all_pages(mock_session, "/users/test/animelist", None)
    ]

    assert len(final_pages) == 3


@pytest.mark.asyncio
async def test_request_page_succesfully_exits_with_blank_page():
    mock_session = SessionStub({})
    conn = MyAnimeListConnection()

    actual = await conn.requests_get_next_page(mock_session, None)

    assert actual is None


@pytest.mark.asyncio
async def test_request_does_not_fail_with_empty_response(mocker):
    response1 = ResponseStub({})

    mocker.patch("animeippo.providers.myanimelist.connection.MAL_API_URL", "FAKE")

    mock_session = SessionStub({"FAKE/test": response1})
    conn = MyAnimeListConnection()

    pages = [page async for page in conn.requests_get_all_pages(mock_session, "/test", None)]

    assert len(pages) == 0


@pytest.mark.asyncio
async def test_request_with_retry_refreshes_token_on_401(mocker):
    unauthorized = ResponseStub({})
    unauthorized.status = HTTP_UNAUTHORIZED

    success = ResponseStub({"data": [{"test": "ok"}]})

    call_count = 0

    class RetrySessionStub:
        def request(self, method, url, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return unauthorized
            return success

    conn = MyAnimeListConnection()
    conn.refresh_token = "fake_refresh"
    mocker.patch.object(conn, "do_token_refresh")

    result = await conn.request_with_retry(RetrySessionStub(), "GET", "http://fake")

    assert result == {"data": [{"test": "ok"}]}
    conn.do_token_refresh.assert_awaited_once()


@pytest.mark.asyncio
async def test_do_token_refresh_updates_tokens(mocker):
    token_response = ResponseStub({"access_token": "new_access", "refresh_token": "new_refresh"})
    mocker.patch("aiohttp.ClientSession.post", return_value=token_response)

    conn = MyAnimeListConnection()
    conn.refresh_token = "old_refresh"
    conn.client_id = "test_id"
    conn.client_secret = "test_secret"

    mocker.patch.object(conn, "persist_tokens")

    await conn.do_token_refresh()

    assert conn.access_token == "new_access"
    assert conn.refresh_token == "new_refresh"
    conn.persist_tokens.assert_called_once()


def test_persist_tokens_writes_to_env_file(mocker):
    mock_set_key = mocker.patch("animeippo.providers.myanimelist.connection.dotenv.set_key")

    conn = MyAnimeListConnection()
    conn.access_token = "test_access"
    conn.refresh_token = "test_refresh"

    conn.persist_tokens()

    assert mock_set_key.call_count == 2
