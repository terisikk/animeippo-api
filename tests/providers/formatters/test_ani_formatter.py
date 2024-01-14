import polars as pl

from animeippo.providers.formatters import ani_formatter
from tests import test_data


def test_tags_can_be_extracted():
    original = pl.DataFrame({"tags": [[{"name": "tag1"}]]})

    assert ani_formatter.get_tags(original).item()[0] == "tag1"


def test_director_can_be_extracted():
    actual = ani_formatter.get_staff(
        [{"role": "Director"}, {"role": "Grunt"}], [{"id": 123}, {"id": 234}], "Director"
    )

    assert actual == ([123],)


def test_dataframe_can_be_constructed_from_ani():
    animelist = {
        "data": test_data.ANI_USER_LIST["data"]["MediaListCollection"]["lists"][0]["entries"]
    }

    data = ani_formatter.transform_watchlist_data(animelist, ["genres", "tags"])

    assert type(data) == pl.DataFrame
    assert "Dr. STRONK: OLD WORLD" in data["title"].to_list()
    assert len(data) == 2


def test_ranks_does_not_break_if_no_ranks():
    tags = {}

    data = ani_formatter.get_temp_ranks(tags)

    assert data != {}
