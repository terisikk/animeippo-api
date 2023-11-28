import pandas as pd
import numpy as np

from animeippo.providers.formatters import mal_formatter
from tests import test_data


def test_genre_splitting_does_not_fail_with_invalid_data():
    mal_formatter.split_id_name_field(None)
    mal_formatter.split_id_name_field(1.0)
    mal_formatter.split_id_name_field([1.0])


def test_user_score_extraction_does_not_fail_with_invalid_data():
    mal_formatter.get_score(1.0)
    mal_formatter.get_score(None)
    mal_formatter.get_score(np.nan)


def test_anime_season_extraction_does_not_fail_with_invalid_data():
    assert mal_formatter.split_season({"year": None, "sesson": "winter"}) == "None/?"
    assert pd.isna(mal_formatter.split_season(np.nan))


def test_user_score_cannot_be_zero():
    original = 0

    actual = mal_formatter.get_score(original)

    assert np.isnan(actual)


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
    assert "coverImage" in data.columns
    assert "genres" in data.columns
