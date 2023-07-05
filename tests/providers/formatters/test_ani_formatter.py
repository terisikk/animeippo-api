import pandas as pd
import numpy as np
import datetime

from animeippo.providers.formatters import ani_formatter
from tests import test_data


class StubMapper:
    def map(self, original):
        return [f"{original} ran"]


def test_tags_can_be_extracted():
    assert ani_formatter.get_tags([{"name": "tag1"}]) == ["tag1"]


def test_season_can_be_extracted():
    assert ani_formatter.get_season(2023, "summer") == "2023/summer"
    assert ani_formatter.get_season(None, "winter") == "?/winter"
    assert ani_formatter.get_season(np.nan, np.nan) == "?/?"


def test_user_score_cannot_be_zero():
    original = 0

    actual = ani_formatter.get_score(original)

    assert pd.isna(actual)


def test_user_complete_date_can_be_extracted():
    actual = ani_formatter.get_user_complete_date(2023, 2, 2)
    assert actual == datetime.date(2023, 2, 2)


def test_dataframe_can_be_constructed_from_ani():
    animelist = {
        "data": test_data.ANI_USER_LIST["data"]["MediaListCollection"]["lists"][0]["entries"]
    }

    data = ani_formatter.transform_watchlist_data(animelist, ["genres", "tags"])

    assert type(data) == pd.DataFrame
    assert data.iloc[0]["title"] == "Dr. STRONK: OLD WORLD"
    assert data.iloc[0]["genres"] == ["Action", "Adventure", "Comedy", "Sci-Fi"]
    assert len(data) == 2


def test_transformation_does_not_fail_with_empty_data():
    data = ani_formatter.transform_to_animeippo_format({}, ["genres", "tags"], [])

    assert type(data) == pd.DataFrame
    assert len(data) == 0

    data = ani_formatter.transform_to_animeippo_format(
        {"data": {"test": "test"}}, ["genres", "tags"], []
    )
    assert type(data) == pd.DataFrame
    assert len(data) == 0


def test_mapping_skips_keys_not_in_dataframe():
    dataframe = pd.DataFrame(columns=["test1", "test2"])
    mapping = {"test1": StubMapper(), "test3": StubMapper()}

    actual = ani_formatter.run_mappers(dataframe, "test1", mapping)
    assert actual["test1"].tolist() == ["test1 ran"]
    assert "test3" not in actual.columns
