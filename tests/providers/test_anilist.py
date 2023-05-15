from animeippo.providers import anilist
from tests import test_data

import requests_mock as rmock


class ResponseStub:
    dictionary = {}

    def __init__(self, dictionary):
        self.dictionary = dictionary

    def json(self):
        return self.dictionary

    def raise_for_status(self):
        pass


def test_ani_user_anime_list_can_be_fetched(requests_mock):
    provider = anilist.AniListProvider()

    user = "Janiskeisari"

    adapter = requests_mock.post(rmock.ANY, json=test_data.ANI_USER_LIST)  # nosec B113

    animelist = provider.get_user_anime_list(user)

    assert adapter.called
    assert "Dr. STONE: NEW WORLD" in animelist["title"].values


def test_ani_seasonal_anime_list_can_be_fetched(requests_mock):
    provider = anilist.AniListProvider()

    year = "2023"
    season = "winter"

    adapter = requests_mock.post(rmock.ANY, json=test_data.ANI_SEASONAL_LIST)  # nosec B113

    animelist = provider.get_seasonal_anime_list(year, season)

    assert adapter.called
    assert "EDENS ZERO 2nd Season" in animelist["title"].values


def test_ani_related_anime_returns_none(requests_mock):
    provider = anilist.AniListProvider()

    animelist = provider.get_related_anime(0)

    assert animelist is None


def test_get_single_returns_succesfully(requests_mock):
    response = {"data": [{"test": "test"}], "pageInfo": {"hasNextPage": False}}

    adapter = requests_mock.post(rmock.ANY, json=response)  # nosec B113

    page = anilist.AnilistConnection().request_single("test", {})

    assert adapter.called
    assert adapter.call_count == 1
    assert page == response


def test_get_all_pages_returns_all_pages(requests_mock):
    response1 = {
        "json": {
            "data": {
                "Page": {
                    "media": {"test": "test2"},
                    "pageInfo": {"hasNextPage": True, "currentPage": 0},
                }
            }
        }
    }
    response2 = {
        "json": {
            "data": {
                "Page": {
                    "media": {"test": "test2"},
                    "pageInfo": {"hasNextPage": True, "currentPage": 1},
                }
            }
        }
    }
    response3 = {
        "json": {
            "data": {
                "Page": {
                    "media": {"test": "test1"},
                    "pageInfo": {"hasNextPage": False, "currentPage": 2},
                }
            }
        }
    }

    adapter = requests_mock.post(rmock.ANY, [response1, response2, response3])  # nosec B113

    final_pages = list(anilist.AnilistConnection().requests_get_all_pages("", {}))

    assert len(final_pages) == 3
    assert final_pages[0] == response1["json"]["data"]["Page"]
    assert final_pages[2] == response3["json"]["data"]["Page"]
    assert adapter.call_count == 3


def test_reqest_does_not_fail_catastrophically_when_response_is_empty(requests_mock):
    response = {}

    adapter = requests_mock.post(rmock.ANY, json=response)  # nosec B113

    all_pages = list(anilist.AnilistConnection().requests_get_all_pages("", {}))

    assert len(all_pages) == 1
    assert all_pages[0] is None
    assert adapter.called


def test_features_can_be_fetched():
    provider = anilist.AniListProvider()

    features = provider.get_features()

    assert len(features) > 0
    assert "genres" in features
