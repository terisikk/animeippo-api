import polars as pl

import animeippo.analysis.similarity


def test_cosine_similarity():
    x_orig = [[True, True, False], [True, False, True]]
    y_orig = [[True, True, False], [True, False, True]]

    distances = animeippo.analysis.similarity.similarity(x_orig, y_orig)

    # Identical vectors have similarity 1.0
    assert f"{distances[0][0]:.1f}" == "1.0"
    # Different vectors have similarity < 1.0
    assert distances[0][1] < 1.0
    assert distances[0][1] > 0.0


def test_categorical_uses_columns_if_given():
    # Create struct series (matching encoder output format)
    original1 = pl.Series([{"a": 1, "b": 2, "c": 3}, {"a": 4, "b": 5, "c": 6}])

    original2 = pl.Series([{"a": 2, "b": 3, "c": 4}, {"a": 1, "b": 2, "c": 3}])

    similarity = animeippo.analysis.similarity.categorical_similarity(
        original1, original2, columns=["1a", "2b"]
    )

    assert similarity.columns == ["1a", "2b"]
