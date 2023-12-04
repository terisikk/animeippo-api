import pandas as pd
import datetime

from animeippo.providers.formatters import ani_formatter
from tests import test_data


def test_tags_can_be_extracted():
    assert ani_formatter.get_tags([{"name": "tag1"}]) == ["tag1"]


def test_user_complete_date_can_be_extracted():
    actual = ani_formatter.get_user_complete_date(2023, 2, 2)
    assert actual == datetime.date(2023, 2, 2)


def test_director_can_be_extracted():
    actual = ani_formatter.get_staff(
        [{"role": "Director"}, {"role": "Grunt"}], [{"id": 123}, {"id": 234}], "Director"
    )

    assert actual == [123]


def test_dataframe_can_be_constructed_from_ani():
    animelist = {
        "data": test_data.ANI_USER_LIST["data"]["MediaListCollection"]["lists"][0]["entries"]
    }

    data = ani_formatter.transform_watchlist_data(animelist, ["genres", "tags"])

    assert type(data) == pd.DataFrame
    assert data.iloc[0]["title"] == "Dr. STRONK: OLD WORLD"
    assert data.iloc[0]["genres"] == ["Action", "Adventure", "Comedy", "Sci-Fi"]
    assert len(data) == 2
