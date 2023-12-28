import pandas as pd

import animeippo.providers.formatters.schema as schema


def test_default_mapper():
    mapper = schema.DefaultMapper("test")

    original = pl.Series({"test": [1, 2, 3]})
    actual = mapper.map(original)

    assert actual == [1, 2, 3]


def test_default_mapper_default_works():
    mapper = schema.DefaultMapper("test")

    original = pl.Series({"wrong": [1, 2, 3]})
    actual = pl.Series(mapper.map(original))

    assert actual.tolist() == [pd.NA]

    mapper = schema.DefaultMapper("test", 123)

    actual = pl.Series(mapper.map(original))

    assert actual.tolist() == [123]


def test_single_mapper():
    mapper = schema.SingleMapper("test", str.lower)

    original = pl.DataFrame({"test": ["TEST1", "Test2", "test3"]})
    actual = mapper.map(original)

    assert actual.to_list() == ["test1", "test2", "test3"]


def test_single_mapper_default_works():
    mapper = schema.SingleMapper("test", str.lower)

    original = pl.DataFrame({"test": ["TEST1", 2, 3]})
    actual = mapper.map(original)

    assert actual.tolist() == ["test1", pd.NA, pd.NA]

    mapper = schema.SingleMapper("test", str.lower, 123)

    original = pl.DataFrame({"wrong": ["TEST1", 2, 3]})
    actual = mapper.map(original)

    assert actual.tolist() == [123, 123, 123]


def test_multi_mapper():
    mapper = schema.MultiMapper(lambda row: row["1"] + row["2"])

    original = pl.DataFrame({"1": [2, 2, 2], "2": [2, 3, 4]})
    actual = mapper.map(original)

    assert actual.to_list() == [4, 5, 6]


def test_multi_mapper_default_works():
    mapper = schema.MultiMapper(lambda row: row["1"] + row["2"])

    original = pl.DataFrame({"2": [2, 2, 2], "3": [2, 3, 4]})
    actual = mapper.map(original)

    assert actual.tolist() == [pd.NA, pd.NA, pd.NA]

    mapper = schema.MultiMapper(lambda row: row["1"] + row["2"], 0)

    original = pl.DataFrame({"1": ["TEST1", 2, 3], "2": [2, 2, 2]})
    actual = mapper.map(original)

    assert actual.tolist() == [0, 4, 5]
