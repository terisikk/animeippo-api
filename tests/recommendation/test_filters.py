import animeippo.recommendation.filters as filters
import polars as pl


def test_media_type_filter():
    original = pl.DataFrame({"format": ["tv", "tv", "special", "movie"]})

    filter = filters.MediaTypeFilter("tv")

    assert original.filter(filter)["format"].to_list() == ["tv", "tv"]

    filter = filters.MediaTypeFilter("tv", negative=True)

    assert original.filter(filter)["format"].to_list() == ["special", "movie"]


def test_feature_filter():
    original = pl.DataFrame({"features": [["Action", "Adventure"], ["Fantasy", "Comedy"]]})

    filter = filters.FeatureFilter("Action")

    assert original.filter(filter)["features"].to_list() == [["Action", "Adventure"]]

    filter = filters.FeatureFilter("Action", negative=True)

    assert original.filter(filter)["features"].to_list() == [["Fantasy", "Comedy"]]


def test_status_filter():
    original = pl.DataFrame(
        {"user_status": ["dropped", "completed", "on_hold", "completed", "unwatched"]}
    )

    filter = filters.UserStatusFilter("completed")

    assert original.filter(filter)["user_status"].to_list() == ["completed", "completed"]

    filter = filters.UserStatusFilter("completed", negative=True)

    assert original.filter(filter)["user_status"].to_list() == ["dropped", "on_hold", "unwatched"]


def test_rating_filter():
    original = pl.DataFrame({"rating": ["g", "r", "g", "pg_13", "r"]})

    filter = filters.RatingFilter("g", "pg_13")

    assert original.filter(filter)["rating"].to_list() == ["g", "g", "pg_13"]

    filter = filters.RatingFilter("g", "pg_13", negative=True)

    assert original.filter(filter)["rating"].to_list() == ["r", "r"]


def test_season_filter():
    original = pl.DataFrame(
        {"season_year": [2023, 2023, 2023], "season": ["winter", "winter", "spring"]}
    )

    filter = filters.StartSeasonFilter([2023], ["winter"])

    assert original.filter(filter)["season"].to_list() == ["winter", "winter"]

    filter = filters.StartSeasonFilter([2023], ["winter"], negative=True)

    assert original.filter(filter)["season"].to_list() == ["spring"]


def test_continuation_filter():
    compare = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "title": ["Anime A", "Anime B", "Anime B Spinoff", "Anime C"],
            "user_status": ["completed", "completed", "completed", "completed"],
        }
    )

    filter = filters.ContinuationFilter(compare)

    original = pl.DataFrame(
        {
            "id": [5, 6, 7, 8],
            "title": ["Anime A Season 2", "Anime E Season 2", "Anime B Season 2", "Anime F"],
            "continuation_to": [[1], [9], [2, 3], []],
        }
    )

    assert original.filter(filter)["id"].to_list() == [5, 7, 8]

    filter = filters.ContinuationFilter(compare, negative=True)

    assert original.filter(filter)["id"].to_list() == [6]
