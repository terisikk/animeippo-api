import polars as pl

from animeippo.recommendation.encoding import CategoricalEncoder, WeightedCategoricalEncoder


def test_categorical_encoded():
    classes = pl.Series(["Test 1", "Test 2", "Test 3"])

    encoder = CategoricalEncoder()
    encoder.fit(classes)

    original = pl.DataFrame({"features": [["Test 3", "Test 2"]]})
    expected = [False, True, True]

    assert encoder.encode(original) == [expected]


def test_weighted_encoder():
    classes = pl.Series(["Test 1", "Test 2", "Test 3"])

    encoder = WeightedCategoricalEncoder()
    encoder.fit(classes)

    original = pl.DataFrame(
        {"features": [["Test 3", "Test 2"]], "ranks": [{"Test 3": 0.85, "Test 2": 0.5}]}
    )
    expected = [0, 0.5, 0.85]

    assert list(encoder.encode(original)[0]) == expected
