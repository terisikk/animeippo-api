import pandas as pd
import numpy as np

from animeippo.providers.formatters import ani_formatter
from tests import test_data


def test_tags_can_be_extracted():
    assert ani_formatter.get_tags([{"name": "tag1"}]) == ["tag1"]

    assert ani_formatter.get_tags([{"malformed": "tag1"}]) == []


def test_season_can_be_extracted():
    assert ani_formatter.format_season({"year": None, "sesson": "winter"}) == "?/?"
    assert ani_formatter.format_season(np.nan) == "?/?"


def test_user_score_cannot_be_zero():
    original = 0

    actual = ani_formatter.formatters["score"](original)

    assert np.isnan(actual)


def test_dataframe_can_be_constructed_from_ani():
    animelist = {
        "data": test_data.ANI_USER_LIST["data"]["MediaListCollection"]["lists"][0]["entries"]
    }

    data = ani_formatter.transform_to_animeippo_format(animelist)

    assert type(data) == pd.DataFrame
    assert data.iloc[0]["title"] == "Dr. STONE: NEW WORLD"
    assert data.iloc[0]["genres"] == ["Action", "Adventure", "Comedy", "Sci-Fi"]
    assert len(data) == 2


def test_transformation_does_not_fail_with_empty_data():
    data = ani_formatter.transform_to_animeippo_format({})

    assert type(data) == pd.DataFrame
    assert len(data) == 0

    data = ani_formatter.transform_to_animeippo_format({"data": {"test": "test"}})
    assert type(data) == pd.DataFrame
    assert len(data) == 1
    assert "id" != data.index.name
