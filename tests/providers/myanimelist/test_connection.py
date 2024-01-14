import pytest
import animeippo.providers.myanimelist.connection

from .stubs import ResponseStub, SessionStub


@pytest.mark.asyncio
async def test_get_next_page_returns_succesfully(mocker):
    response1 = ResponseStub({"data": [{"test": "test"}], "paging": {"next": "page2"}})
    response2 = ResponseStub({"data": [{"test2": "test2"}], "paging": {"next": "page3"}})
    response3 = ResponseStub({"data": [{"test3": "test3"}]})

    pages = [response1, response2, response3]

    mocker.patch("aiohttp.ClientSession.get", side_effect=pages)

    session = SessionStub({"page1": response1, "page2": response2, "page3": response3})
    final_pages = [
        await animeippo.providers.myanimelist.MyAnimeListConnection().requests_get_next_page(
            session, await page.json()
        )
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

    mocker.patch("animeippo.providers.myanimelist.connection.MAL_API_URL", "FAKE")
    first_page_url = "/users/kamina69/animelist"

    response = ResponseStub({"related_anime": []})
    mocker.patch("aiohttp.ClientSession.get", return_value=response)

    mock_session = SessionStub(
        {"FAKE" + first_page_url: response1, "page2": response2, "page3": response3}
    )

    final_pages = list(
        [
            page
            async for page in animeippo.providers.myanimelist.MyAnimeListConnection().requests_get_all_pages(
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

    actual = await animeippo.providers.myanimelist.MyAnimeListConnection().requests_get_next_page(
        mock_session, page
    )

    assert actual is None


@pytest.mark.asyncio
async def test_request_does_not_fail_catastrophically_when_response_is_empty(mocker):
    response1 = ResponseStub(dict())

    mocker.patch("animeippo.providers.myanimelist.connection.MAL_API_URL", "FAKE")
    first_page_url = "/users/kamina69/animelist"

    mock_session = SessionStub({"FAKE" + first_page_url: response1})

    pages = list(
        [
            page
            async for page in animeippo.providers.myanimelist.MyAnimeListConnection().requests_get_all_pages(
                mock_session, first_page_url, None
            )
        ]
    )

    assert len(pages) == 0
