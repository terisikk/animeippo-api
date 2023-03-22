import pytest

from animeippo import myanimelist


class SessionStub:
    dictionary = {}

    def __init__(self, dictionary):
        self.dictionary = dictionary

    def get(self, key, *args, **kwargs):
        return self.dictionary.get(key)


class ResponseStub:
    dictionary = {}

    def __init__(self, dictionary):
        self.dictionary = dictionary

    def json(self):
        return self.dictionary

    def raise_for_status(self):
        pass


def test_mal_anime_list_can_be_fetched():
    user = "Janiskeisari"
    animelist = myanimelist.get_anime_list(user)
    assert "91 Days" in [node["title"] for node in animelist]


def test_get_next_page_returns_succesfully():
    response1 = ResponseStub({"data": [{"test": "test"}], "paging": {"next": "page2"}})
    response2 = ResponseStub({"data": [{"test2": "test2"}], "paging": {"next": "page3"}})
    response3 = ResponseStub({"data": [{"test3": "test3"}]})

    pages = [response1, response2, response3]

    mock_session = SessionStub({"page1": response1, "page2": response2, "page3": response3})

    final_pages = [myanimelist.requests_get_next_page(mock_session, page.json()) for page in pages]

    assert len(final_pages) == 3
    assert final_pages[0] == response2.json()
    assert final_pages[1] == response3.json()
    assert final_pages[2] == None


def test_get_all_pages_returns_all_pages():
    response1 = ResponseStub({"data": [{"test": "test"}], "paging": {"next": "page2"}})
    response2 = ResponseStub({"data": [{"test2": "test2"}], "paging": {"next": "page3"}})
    response3 = ResponseStub({"data": [{"test3": "test3"}]})

    myanimelist.MAL_API_URL = "FAKE"
    first_page_url = "FAKE/users/kamina69/animelist"

    mock_session = SessionStub({first_page_url: response1, "page2": response2, "page3": response3})

    final_pages = list(myanimelist.requests_get_all_pages(mock_session, "kamina69"))

    assert len(final_pages) == 3
    assert final_pages[0] == response1.json()
    assert final_pages[1] == response2.json()
    assert final_pages[2] == response3.json()


def test_reqest_does_not_fail_catastrophically_when_response_is_empty():
    response1 = ResponseStub(dict())

    myanimelist.MAL_API_URL = "FAKE"
    first_page_url = "FAKE/users/kamina69/animelist"

    mock_session = SessionStub({first_page_url: response1})

    pages = list(myanimelist.requests_get_all_pages(mock_session, "kamina69"))

    assert len(pages) == 0
