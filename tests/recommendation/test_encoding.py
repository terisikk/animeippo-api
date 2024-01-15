import polars as pl

from animeippo.analysis.encoding import CategoricalEncoder, WeightedCategoricalEncoder


def test_categorical_encoder():
    classes = pl.Series(["Test 1", "Test 2", "Test 3"])

    encoder = CategoricalEncoder()
    encoder.fit(classes)

    original = pl.DataFrame({"features": [["Test 3", "Test 2"]]})
    expected = [False, True, True]

    assert encoder.encode(original).tolist() == [expected]


def test_weighted_encoder():
    classes = pl.Series(["Test 1", "Test 2", "Test 3"])

    encoder = WeightedCategoricalEncoder()
    encoder.fit(classes)

    original = pl.DataFrame({"features": [["Test 3", "Test 2"]], "ranks": [[0.85, 0.5]]})
    expected = [0, 0.5, 0.85]

    actual = encoder.encode(original)

    assert list(actual[0]) == expected
