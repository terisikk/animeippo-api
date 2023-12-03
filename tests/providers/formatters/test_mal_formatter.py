import pandas as pd
import numpy as np

from animeippo.providers.formatters import mal_formatter, util
from tests import test_data


def test_dataframe_can_be_constructed_from_mal():
    animelist = test_data.MAL_USER_LIST

    data = mal_formatter.transform_watchlist_data(animelist, ["genres"])

    assert type(data) == pd.DataFrame
    assert len(data) == 2
    assert data.iloc[1]["title"] == "Hellsingfårs"
    assert data.iloc[1]["genres"] == [
        "Action",
        "Adult Cast",
        "Gore",
        "Horror",
        "Seinen",
        "Supernatural",
        "Vampire",
    ]
    assert data.iloc[1]["score"] == 8


def test_transform_does_not_fail_on_missing_id_column():
    data = ["test1", "test2"]
    feature_names = []
    keys = []

    actual = util.transform_to_animeippo_format(
        data, feature_names, keys, mal_formatter.MAL_MAPPING
    )

    assert "id" not in actual.columns
    assert actual.index.name != "id"


def test_relations_can_be_constructed_from_mal():
    animelist = {"data": test_data.MAL_RELATED_ANIME["related_anime"]}

    data = mal_formatter.transform_related_anime(animelist, [])

    assert len(data) == 1
    assert data == [31]


def test_dataframe_can_be_constructed_from_incomplete_data():
    animelist = test_data.MAL_USER_LIST

    del animelist["data"][0]["list_status"]
    del animelist["data"][1]["list_status"]

    data = mal_formatter.transform_watchlist_data(animelist, ["genres"])

    assert type(data) == pd.DataFrame
    assert len(data) == 2
    assert data.iloc[1]["title"] == "Hellsingfårs"
    assert data.iloc[1]["genres"] == [
        "Action",
        "Adult Cast",
        "Gore",
        "Horror",
        "Seinen",
        "Supernatural",
        "Vampire",
    ]
    assert pd.isnull(data.iloc[1].get("score", np.nan))
    assert "user_status" in data.columns


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

    actual = mal_formatter.split_id_name_field(original)

    expected = ["Action", "Adult Cast", "Gore", "Horror", "Seinen", "Supernatural", "Vampire"]

    assert actual == expected


def test_columns_are_named_properly():
    animelist = test_data.MAL_SEASONAL_LIST

    data = mal_formatter.transform_seasonal_data(animelist, [])

    assert "popularity" in data.columns
    assert "cover_image" in data.columns
    assert "genres" in data.columns


def test_get_continuation():
    relation = "prequel"
    id = 123

    assert mal_formatter.get_continuation(relation, id) == 123

    relation = "irrelevant"

    assert pd.isna(mal_formatter.get_continuation(relation, id))


def test_get_image_url():
    field = {"medium": "test"}

    assert mal_formatter.get_image_url(field) == "test"

    assert pd.isna(mal_formatter.get_image_url({}))


def test_get_user_complete_date():
    assert not pd.isna(mal_formatter.get_user_complete_date("2020-03-12"))

    assert pd.isna(mal_formatter.get_user_complete_date(pd.NA))


def test_get_status():
    assert mal_formatter.get_status("currently_airing") == "releasing"

    assert mal_formatter.get_status("invalid") == "invalid"
