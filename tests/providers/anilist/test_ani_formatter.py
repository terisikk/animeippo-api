import polars as pl

from animeippo.providers.anilist.provider import formatter
from tests import test_data


def test_tags_can_be_extracted():
    original = pl.DataFrame({"tags": [[{"name": "tag1"}]]})

    assert formatter.get_tags(original).item()[0] == "tag1"


def test_dataframe_can_be_constructed_from_ani():
    animelist = {
        "data": test_data.ANI_USER_LIST["data"]["MediaListCollection"]["lists"][0]["entries"]
    }

    data = formatter.transform_watchlist_data(animelist, ["genres", "tags"])

    assert type(data) is pl.DataFrame
    assert "Dr. STRONK: OLD WORLD" in data["title"].to_list()
    assert len(data) == 2
