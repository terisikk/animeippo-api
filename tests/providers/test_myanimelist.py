import pandas as pd
import numpy as np

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
    assert final_pages[2] is None


def test_get_all_pages_returns_all_pages(mocker):
    response1 = ResponseStub({"data": [{"test": "test"}], "paging": {"next": "page2"}})
    response2 = ResponseStub({"data": [{"test2": "test2"}], "paging": {"next": "page3"}})
    response3 = ResponseStub({"data": [{"test3": "test3"}]})

    mocker.patch("animeippo.providers.myanimelist.MAL_API_URL", "FAKE")
    first_page_url = "FAKE/users/kamina69/animelist"

    mock_session = SessionStub({first_page_url: response1, "page2": response2, "page3": response3})

    final_pages = list(myanimelist.requests_get_all_pages(mock_session, first_page_url, None))

    assert len(final_pages) == 3
    assert final_pages[0] == response1.json()


def test_request_page_succesfully_exists_with_blank_page():
    page = None
    mock_session = SessionStub({})

    actual = myanimelist.requests_get_next_page(mock_session, page)

    assert actual is None


def test_reqest_does_not_fail_catastrophically_when_response_is_empty(mocker):
    response1 = ResponseStub(dict())

    mocker.patch("animeippo.providers.myanimelist.MAL_API_URL", "FAKE")
    first_page_url = "FAKE/users/kamina69/animelist"

    mock_session = SessionStub({first_page_url: response1})

    pages = list(myanimelist.requests_get_all_pages(mock_session, first_page_url, None))

    assert len(pages) == 0


def test_mal_genres_can_be_split():
    original = [
        {"id": 1, "name": "Action"},
        {"id": 50, "name": "Adult Cast"},
        {"id": 58, "name": "Gore"},
        {"id": 14, "name": "Horror"},
        {"id": 42, "name": "Seinen"},
        {"id": 37, "name": "Supernatural"},
        {"id": 32, "name": "Vampire"},
    ]

    actual = myanimelist.split_id_name_field(original)

    expected = ["Action", "Adult Cast", "Gore", "Horror", "Seinen", "Supernatural", "Vampire"]

    assert actual == expected


def test_genre_splitting_does_not_fail_with_invalid_data():
    myanimelist.split_id_name_field(None)
    myanimelist.split_id_name_field(1.0)
    myanimelist.split_id_name_field([1.0])


def test_user_score_can_be_extracted():
    original = {"score": 10, "status": "completed"}

    expected = 10
    actual = myanimelist.get_user_score(original)

    assert actual == expected


def test_user_score_extraction_does_not_fail_with_invalid_data():
    myanimelist.get_user_score(1.0)
    myanimelist.get_user_score(None)


def test_anime_status_can_be_extracted():
    original = {"score": 10, "status": "completed"}

    expected = "completed"
    actual = myanimelist.get_user_anime_status(original)

    assert actual == expected


def test_anime_status_extraction_does_not_fail_with_invalid_data():
    myanimelist.get_user_anime_status(1.0)
    myanimelist.get_user_anime_status(None)


def test_user_score_cannot_be_zero():
    original = {"score": 0, "status": "completed"}
    expected = np.nan

    actual = myanimelist.get_user_score(original)

    assert actual is expected


def test_dataframe_can_be_constructed_from_mal():
    provider = myanimelist.MyAnimeListProvider()

    animelist = [item for item in test_data.MAL_USER_LIST["data"]]

    data = provider.transform_to_animeippo_format(animelist)

    assert type(data) == pd.DataFrame
    assert len(data) == 2
    assert data.loc[1, "title"] == "Hellsing"
    assert data.loc[1, "genres"] == [
        "Action",
        "Adult Cast",
        "Gore",
        "Horror",
        "Seinen",
        "Supernatural",
        "Vampire",
    ]
    assert data.loc[1, "user_score"] == 8


def test_dataframe_can_be_constructed_from_incomplete_data():
    provider = myanimelist.MyAnimeListProvider()

    animelist = [item for item in test_data.MAL_USER_LIST["data"]]

    del animelist[0]["list_status"]
    del animelist[1]["list_status"]

    data = provider.transform_to_animeippo_format(animelist)

    assert type(data) == pd.DataFrame
    assert len(data) == 2
    assert data.loc[1, "title"] == "Hellsing"
    assert data.loc[1, "genres"] == [
        "Action",
        "Adult Cast",
        "Gore",
        "Horror",
        "Seinen",
        "Supernatural",
        "Vampire",
    ]
    assert pd.isnull(data.loc[1, "user_score"])
    assert "list_status" not in data.columns


def test_asdf():
    asdf = np.nan

    assert pd.isnull(asdf)
    assert pd.isnull(np.nan)
    assert pd.isnull(None)
