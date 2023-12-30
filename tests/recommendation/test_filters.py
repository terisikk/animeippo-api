import animeippo.recommendation.filters as filters
import polars as pl


def test_abstract_filter_can_be_instantiated():
    class ConcreteFilter(filters.AbstractFilter):
        def filter(self, dataframe):
            return super().filter(dataframe)

    filter = ConcreteFilter()
    filter.filter(None)

    assert issubclass(filter.__class__, filters.AbstractFilter)


def test_media_type_filter():
    original = pl.DataFrame({"format": ["tv", "tv", "special", "movie"]})

    filter = filters.MediaTypeFilter("tv")

    assert filter.filter(original)["format"].to_list() == ["tv", "tv"]

    filter.negative = True

    assert filter.filter(original)["format"].to_list() == ["special", "movie"]


def test_feature_filter():
    original = pl.DataFrame({"features": [["Action", "Adventure"], ["Fantasy", "Comedy"]]})

    filter = filters.FeatureFilter("Action")

    assert filter.filter(original)["features"].to_list() == [["Action", "Adventure"]]

    filter.negative = True

    assert filter.filter(original)["features"].to_list() == [["Fantasy", "Comedy"]]


def test_status_filter():
    original = pl.DataFrame(
        {"user_status": ["dropped", "completed", "on_hold", "completed", "unwatched"]}
    )

    filter = filters.UserStatusFilter("completed")

    assert filter.filter(original)["user_status"].to_list() == ["completed", "completed"]

    filter.negative = True

    assert filter.filter(original)["user_status"].to_list() == ["dropped", "on_hold", "unwatched"]


def test_rating_filter():
    original = pl.DataFrame({"rating": ["g", "r", "g", "pg_13", "r"]})

    filter = filters.RatingFilter("g", "pg_13")

    assert filter.filter(original)["rating"].to_list() == ["g", "g", "pg_13"]

    filter.negative = True

    assert filter.filter(original)["rating"].to_list() == ["r", "r"]


def test_season_filter():
    original = pl.DataFrame({"start_season": ["2023/winter", "2023/winter", "2023/spring"]})

    filter = filters.StartSeasonFilter(("2023", "winter"))

    assert filter.filter(original)["start_season"].to_list() == ["2023/winter", "2023/winter"]

    filter.negative = True

    assert filter.filter(original)["start_season"].to_list() == ["2023/spring"]


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

    assert filter.filter(original)["id"].to_list() == [5, 7, 8]

    filter.negative = True

    assert filter.filter(original)["id"].to_list() == [6]
