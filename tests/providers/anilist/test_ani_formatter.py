import polars as pl

from animeippo.providers.anilist import data as anilist_data
from animeippo.providers.anilist.provider import formatter
from tests import test_data


def test_dataframe_can_be_constructed_from_ani():
    animelist = {
        "data": test_data.ANI_USER_LIST["data"]["MediaListCollection"]["lists"][1]["entries"]
    }

    # Use tag lookup from static data (already in dict format with IDs as keys)
    tag_lookup = anilist_data.ALL_TAGS

    data = formatter.transform_watchlist_data(animelist, ["genres", "tags"], tag_lookup)

    assert type(data) is pl.DataFrame
    assert "Dr. STRONK: OLD WORLD" in data["title"].to_list()
    assert len(data) == 2
