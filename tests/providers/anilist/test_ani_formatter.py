import polars as pl

from animeippo.providers.anilist import data as anilist_data
from animeippo.providers.anilist.provider import formatter
from tests import test_data

FEATURE_FIELDS = ["genres", "tags"]


def test_dataframe_can_be_constructed_from_ani():
    animelist = {
        "data": test_data.ANI_USER_LIST["data"]["MediaListCollection"]["lists"][1]["entries"]
    }

    tag_lookup = anilist_data.ALL_TAGS

    data = formatter.transform_watchlist_data(animelist, FEATURE_FIELDS, tag_lookup)

    assert type(data) is pl.DataFrame
    assert "Dr. STRONK: OLD WORLD" in data["title"].to_list()
    assert len(data) == 2


def test_franchise_column_is_built_from_relations():
    animelist = {
        "data": test_data.ANI_USER_LIST["data"]["MediaListCollection"]["lists"][1]["entries"]
    }

    tag_lookup = anilist_data.ALL_TAGS

    data = formatter.transform_watchlist_data(animelist, FEATURE_FIELDS, tag_lookup)

    assert "franchise" in data.columns
    # Both entries are related (SEQUEL/PREQUEL), so both should have the same franchise
    franchises = data["franchise"].to_list()
    assert len(franchises[0]) == 1
    assert franchises[0] == franchises[1]


def test_get_staff_extracts_directors():
    original = pl.DataFrame(
        {
            "id": [1, 2],
            "staff.edges": [
                [{"role": "Director"}, {"role": "Animation Director"}],
                [{"role": "Script"}, {"role": "Director"}],
            ],
            "staff.nodes": [
                [{"id": 100}, {"id": 200}],
                [{"id": 300}, {"id": 400}],
            ],
        }
    )

    result = formatter.get_staff(original)

    assert result[0].to_list() == [100]
    assert result[1].to_list() == [400]
