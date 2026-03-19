import polars as pl

import animeippo.analysis.similarity


def test_jaccard_similarity():
    x_orig = [[True, True, False], [True, False, True]]
    y_orig = [[True, True, False], [True, False, True]]

    distances = animeippo.analysis.similarity.similarity(x_orig, y_orig)

    expected0 = "1.0"
    actual0 = f"{distances[0][0]:.1f}"

    expected1 = "0.3"
    actual1 = f"{distances[0][1]:.1f}"

    assert actual0 == expected0
    assert actual1 == expected1


def test_categorical_uses_columns_if_given():
    # Create struct series (matching encoder output format)
    original1 = pl.Series([{"a": 1, "b": 2, "c": 3}, {"a": 4, "b": 5, "c": 6}])

    original2 = pl.Series([{"a": 2, "b": 3, "c": 4}, {"a": 1, "b": 2, "c": 3}])

    similarity = animeippo.analysis.similarity.categorical_similarity(
        original1, original2, columns=["1a", "2b"]
    )

    assert similarity.columns == ["1a", "2b"]
