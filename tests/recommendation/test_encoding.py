import polars as pl

from animeippo.analysis.encoding import CategoricalEncoder, WeightedCategoricalEncoder


def test_categorical_encoder():
    classes = pl.Series(["Test 1", "Test 2", "Test 3"])

    encoder = CategoricalEncoder()
    encoder.fit(classes)

    original = pl.DataFrame({"features": [["Test 3", "Test 2"]]})

    actual = encoder.encode(original).struct.unnest().to_numpy()[0]
    assert actual.tolist() == [0, 1, 1]


def test_weighted_encoder():
    classes = pl.Series(["Test 1", "Test 2", "Test 3"])

    encoder = WeightedCategoricalEncoder()
    encoder.fit(classes)

    original = pl.DataFrame(
        {"features": [["Test 3", "Test 2"]], "clustering_ranks": [{"Test 3": 85, "Test 2": 50}]}
    )

    actual = encoder.encode(original).struct.unnest().fill_null(0).to_numpy()[0]
    assert actual.tolist() == [0, 50, 85]
