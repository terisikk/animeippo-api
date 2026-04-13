import polars as pl

from animeippo.providers import util


class StubMapper:
    def map(self, original):
        return pl.Series([f"{original} ran"])


def test_mapping_skips_keys_not_in_dataframe():
    mapping = {"test1": StubMapper(), "test3": StubMapper()}

    actual = util.run_mappers("test1", mapping, {"test1": pl.Utf8})
    assert actual["test1"].to_list() == ["test1 ran"]
    assert "test3" not in actual.columns


def test_build_franchise_ids_groups_related_anime():
    ids = pl.Series("id", [1, 2, 3, 4, 5])
    relations = pl.Series(
        "rels",
        [
            [2],  # 1 → 2
            [1, 3],  # 2 → 1, 3
            [2],  # 3 → 2
            [5],  # 4 → 5
            [4],  # 5 → 4
        ],
    )

    result = util.build_franchise_ids(ids, relations).to_list()

    # 1, 2, 3 should share a franchise; 4, 5 should share a different one
    assert result[0] == result[1] == result[2]
    assert result[3] == result[4]
    assert result[0] != result[3]


def test_build_franchise_ids_singletons_get_empty_list():
    ids = pl.Series("id", [1, 2, 3])
    relations = pl.Series(
        "rels",
        [
            [2],  # 1 → 2
            [1],  # 2 → 1
            [],  # 3 has no relations
        ],
    )

    result = util.build_franchise_ids(ids, relations)

    assert len(result[0]) == 1  # franchise member
    assert len(result[1]) == 1  # franchise member
    assert len(result[2]) == 0  # singleton


def test_build_franchise_ids_external_relations_dont_form_franchise():
    """Anime related only to items outside the watchlist are singletons."""
    ids = pl.Series("id", [1, 2])
    relations = pl.Series(
        "rels",
        [
            [99],  # 1 → 99 (not in watchlist)
            [100],  # 2 → 100 (not in watchlist)
        ],
    )

    result = util.build_franchise_ids(ids, relations)

    assert len(result[0]) == 0
    assert len(result[1]) == 0


def test_clustering_ranks_with_only_genres():
    df = pl.DataFrame(
        {
            "id": [1],
            "feature_info": [
                [
                    {
                        "name": "Action",
                        "rank": 100,
                        "category": "Genre",
                        "mood": "hype",
                        "intensity": None,
                    }
                ]
            ],
        }
    )
    result = util.get_clustering_ranks(df)
    assert len(result) == 1


def test_clustering_ranks_with_only_tags():
    df = pl.DataFrame(
        {
            "id": [1],
            "feature_info": [
                [
                    {
                        "name": "Gore",
                        "rank": 85,
                        "category": "Theme-Other",
                        "mood": "dark",
                        "intensity": "heavy",
                    }
                ]
            ],
        }
    )
    result = util.get_clustering_ranks(df)
    assert len(result) == 1


def test_transformation_does_not_fail_with_empty_data():
    data = util.transform_to_animeippo_format(pl.DataFrame(), [], {})

    assert type(data) is pl.DataFrame
    assert len(data) == 0

    data = util.transform_to_animeippo_format(pl.DataFrame({"data": {"test": "test"}}), {}, {})
    assert type(data) is pl.DataFrame
    assert len(data) == 0
