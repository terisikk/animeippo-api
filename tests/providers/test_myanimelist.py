from animeippo.providers import myanimelist
from tests import test_data


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


def test_mal_user_anime_list_can_be_fetched(requests_mock):
    provider = myanimelist.MyAnimeListProvider()

    user = "Janiskeisari"

    url = f"{myanimelist.MAL_API_URL}/users/{user}/animelist"
    adapter = requests_mock.get(url, json=test_data.MAL_USER_LIST)  # nosec B113

    animelist = provider.get_user_anime_list(user)

    assert adapter.called
    assert "Hellsing" in animelist["title"].values


def test_mal_seasonal_anime_list_can_be_fetched(requests_mock):
    provider = myanimelist.MyAnimeListProvider()

    year = "2023"
    season = "winter"

    url = f"{myanimelist.MAL_API_URL}/anime/season/{year}/{season}"
    adapter = requests_mock.get(url, json=test_data.MAL_SEASONAL_LIST)  # nosec B113

    animelist = provider.get_seasonal_anime_list(year, season)

    assert adapter.called
    assert "Shingeki no Kyojin: The Final Season" in animelist["title"].values


def test_mal_related_anime_can_be_fetched(requests_mock):
    provider = myanimelist.MyAnimeListProvider()

    anime_id = 30

    url = f"{myanimelist.MAL_API_URL}/anime/{anime_id}"
    adapter = requests_mock.get(url, json=test_data.MAL_RELATED_ANIME)  # nosec B113

    details = provider.get_related_anime(anime_id)

    assert adapter.called
    assert details.index.tolist() == [31]


def test_mal_related_anime_does_not_fail_with_invalid_data(requests_mock):
    provider = myanimelist.MyAnimeListProvider()

    anime_id = 30

    url = f"{myanimelist.MAL_API_URL}/anime/{anime_id}"
    adapter = requests_mock.get(url, json={"related_anime": []})  # nosec B113

    details = provider.get_related_anime(anime_id)

    assert adapter.called
    assert details.index.tolist() == []


def test_get_next_page_returns_succesfully():
    response1 = ResponseStub({"data": [{"test": "test"}], "paging": {"next": "page2"}})
    response2 = ResponseStub({"data": [{"test2": "test2"}], "paging": {"next": "page3"}})
    response3 = ResponseStub({"data": [{"test3": "test3"}]})

    pages = [response1, response2, response3]

    mock_session = SessionStub({"page1": response1, "page2": response2, "page3": response3})

    final_pages = [
        myanimelist.MyAnimeListProvider().requests_get_next_page(mock_session, page.json())
        for page in pages
    ]

    assert len(final_pages) == 3
    assert final_pages[0] == response2.json()
    assert final_pages[1] == response3.json()
    assert final_pages[2] is None


def test_get_all_pages_returns_all_pages(mocker):
    response1 = ResponseStub({"data": [{"test": "test"}], "paging": {"next": "page2"}})
    response2 = ResponseStub({"data": [{"test2": "test2"}], "paging": {"next": "page3"}})
    response3 = ResponseStub({"data": [{"test3": "test3"}]})

    mocker.patch("animeippo.providers.myanimelist.MAL_API_URL", "FAKE")
    first_page_url = "/users/kamina69/animelist"

    mock_session = SessionStub(
        {"FAKE" + first_page_url: response1, "page2": response2, "page3": response3}
    )

    final_pages = list(
        myanimelist.MyAnimeListProvider().requests_get_all_pages(mock_session, first_page_url, None)
    )

    assert len(final_pages) == 3
    assert final_pages[0] == response1.json()


def test_request_page_succesfully_exists_with_blank_page():
    page = None
    mock_session = SessionStub({})

    actual = myanimelist.MyAnimeListProvider().requests_get_next_page(mock_session, page)

    assert actual is None


def test_reqest_does_not_fail_catastrophically_when_response_is_empty(mocker):
    response1 = ResponseStub(dict())

    mocker.patch("animeippo.providers.myanimelist.MAL_API_URL", "FAKE")
    first_page_url = "/users/kamina69/animelist"

    mock_session = SessionStub({"FAKE" + first_page_url: response1})

    pages = list(
        myanimelist.MyAnimeListProvider().requests_get_all_pages(mock_session, first_page_url, None)
    )

    assert len(pages) == 0


def test_genre_tags_can_be_fetched():
    provider = myanimelist.MyAnimeListProvider()

    genre_tags = provider.get_genre_tags()

    assert len(genre_tags) > 0
    assert "Action" in genre_tags
