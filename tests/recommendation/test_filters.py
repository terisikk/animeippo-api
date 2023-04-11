import animeippo.recommendation.filters as filters
import pandas as pd


def test_abstract_filter_can_be_instantiated():
    class ConcreteFilter(filters.AbstractFilter):
        def filter(self, dataframe):
            return super().filter(dataframe)

    filter = ConcreteFilter()
    filter.filter(None)

    assert issubclass(filter.__class__, filters.AbstractFilter)


def test_media_type_filter():
    original = pd.DataFrame({"media_type": ["tv", "tv", "special", "movie"]})

    filter = filters.MediaTypeFilter("tv")

    assert filter.filter(original)["media_type"].tolist() == ["tv", "tv"]

    filter.negative = True

    assert filter.filter(original)["media_type"].tolist() == ["special", "movie"]


def test_genre_filter():
    original = pd.DataFrame({"genres": [["Action", "Adventure"], ["Fantasy", "Comedy"]]})

    filter = filters.GenreFilter("Action")

    assert filter.filter(original)["genres"].tolist() == [["Action", "Adventure"]]

    filter.negative = True

    assert filter.filter(original)["genres"].tolist() == [["Fantasy", "Comedy"]]


def test_id_filter():
    original = pd.DataFrame({"id": [1, 2, 3, 4], "data": ["a", "b", "c", "d"]})
    original = original.set_index("id")

    filter = filters.IdFilter(1, 3)

    assert filter.filter(original).index.tolist() == [1, 3]

    filter.negative = True

    assert filter.filter(original).index.tolist() == [2, 4]


def test_status_filter():
    original = pd.DataFrame(
        {"status": ["dropped", "completed", "on_hold", "completed", "unwatched"]}
    )

    filter = filters.StatusFilter("completed")

    assert filter.filter(original)["status"].tolist() == ["completed", "completed"]

    filter.negative = True

    assert filter.filter(original)["status"].tolist() == ["dropped", "on_hold", "unwatched"]


def test_filters_work_with_lists():
    original = pd.DataFrame({"id": [1, 2, 3, 4]})

    filter = filters.IdFilter(*[1, 3])

    assert filter.filter(original)["id"].tolist() == [1, 3]

    filter.negative = True

    assert filter.filter(original)["id"].tolist() == [2, 4]
