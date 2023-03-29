import json

from tests import test_data
from animeippo.providers import animeofflinedb
from unittest import mock
from ijson.compat import utf8reader


def test_find_all_similar_anime():
    mock_json = json.dumps(test_data.ANIME_OFFLINE_DATA)

    with mock.patch("builtins.open", mock.mock_open(read_data=mock_json)):
        # Throws deprecation warning otherwise, and might break in future versions
        with mock.patch("ijson.compat.bytes_reader", lambda f: utf8reader(f)):
            actual = animeofflinedb.find_all_similar_anime(["bears"])

            assert len(actual) == 1
            assert "genres" in actual.columns
            assert "sources" not in actual.columns
            assert "bears" in actual.at[0, "genres"]


def test_find_by_titles():
    mock_json = json.dumps(test_data.ANIME_OFFLINE_DATA)

    with mock.patch("builtins.open", mock.mock_open(read_data=mock_json)):
        # Throws deprecation warning otherwise, and might break in future versions
        with mock.patch("ijson.compat.bytes_reader", lambda f: utf8reader(f)):
            title = "Ginga Nagareboshi Gin"
            actual = animeofflinedb.find_by_titles([title])

            assert len(actual) == 1
            assert "genres" in actual.columns
            assert "sources" not in actual.columns
            assert title in actual["title"].values
