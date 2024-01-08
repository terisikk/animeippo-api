import math
import polars as pl

import animeippo.providers.formatters.schema as schema


def test_default_mapper():
    mapper = schema.DefaultMapper("test")

    original = pl.DataFrame({"test": [1, 2, 3]})
    actual = mapper.map(original)

    assert actual.to_list() == [1, 2, 3]


def test_default_mapper_default_works():
    original = pl.DataFrame({"wrong": [1, 2, 3]})

    mapper = schema.DefaultMapper("test", 123)
    actual = pl.DataFrame().with_columns(test=mapper.map(original))

    assert actual["test"].to_list() == [123]


def test_single_mapper():
    mapper = schema.SingleMapper("test", str.lower)

    original = pl.DataFrame({"test": ["TEST1", "Test2", "test3"]})
    actual = mapper.map(original)

    assert actual.to_list() == ["test1", "test2", "test3"]


def test_single_mapper_default_works():
    mapper = schema.SingleMapper("test", str.lower)

    original = pl.DataFrame({"test": pl.Series(["TEST1", 2, 3])})
    actual = pl.DataFrame().with_columns(test=mapper.map(original))

    assert actual["test"].to_list() == ["test1", None, None]

    mapper = schema.SingleMapper("test", str.lower, 123)

    original = pl.DataFrame({"wrong": ["TEST1", 2, 3]})
    actual = pl.DataFrame({"existing": pl.Series([1, 2, 3])}).with_columns(
        test=mapper.map(original)
    )

    assert actual["test"].to_list() == [123, 123, 123]

    mapper = schema.SingleMapper("test", math.pow, 5)

    original = pl.DataFrame({"test": [1, 2, 3]})
    actual = pl.DataFrame().with_columns(test=mapper.map(original))

    assert actual["test"].to_list() == [5, 5, 5]


def test_multi_mapper():
    mapper = schema.MultiMapper(["1", "2"], lambda x, y: x + y)

    original = pl.DataFrame({"1": [2, 2, 2], "2": [2, 3, 4]})
    actual = pl.Series(mapper.map(original))

    assert actual.to_list() == [4, 5, 6]


def test_multi_mapper_default_works():
    mapper = schema.MultiMapper(["1", "2"], lambda x, y: x + y)

    original = pl.DataFrame({"2": [2, 2, 2], "3": [2, 3, 4]})
    actual = pl.DataFrame().with_columns(test=mapper.map(original))

    assert actual["test"].to_list() == [None]

    mapper = schema.MultiMapper(["1", "2"], lambda x, y: x + y, 0)

    original = pl.DataFrame({"1": [None, 2, 3], "2": [2, 2, 2]})
    actual = pl.DataFrame().with_columns(test=mapper.map(original))

    assert actual["test"].to_list() == [0, 4, 5]
