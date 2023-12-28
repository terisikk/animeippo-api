import animeippo.recommendation.analysis as analysis
import pandas as pd
import numpy as np
import pytest
import polars as pl


class EncoderStub:
    def encode(self, values):
        return [np.array(value) for value in values]


def test_jaccard_similarity():
    x_orig = [[True, True, False], [True, False, True]]
    y_orig = [[True, True, False], [True, False, True]]

    distances = analysis.similarity(x_orig, y_orig)

    expected0 = "1.0"
    actual0 = "{:.1f}".format(distances[0][0])

    expected1 = "0.3"
    actual1 = "{:.1f}".format(distances[0][1])

    assert actual0 == expected0
    assert actual1 == expected1


def test_genre_average_scores():
    original = pl.DataFrame(
        {
            "genres": [["Action"], ["Action", "Horror"], ["Action", "Horror", "Romance"]],
            "score": [10, 10, 7],
        }
    )

    avg = analysis.mean_score_per_categorical(original.explode("genres"), "genres")

    assert avg.sort("genres")["score"].to_list() == [9.0, 8.5, 7.0]


def test_similarity_weights():
    genre_averages = pl.DataFrame([("Action", 9.0), ("Horror", 8.0), ("Romance", 7.0)])
    genre_averages.columns = ["genres", "weight"]

    original = pl.DataFrame(
        {
            "genres": [["Action", "Horror"], ["Action", "Romance"]],
        }
    )

    weights = original["genres"].apply(
        lambda row: analysis.weighted_mean_for_categorical_values(row, genre_averages)
    )

    assert weights.sort(descending=True).to_list() == [8.5, 8.0]


def test_similarity_weight_uses_zero_to_subsitute_nan():
    genre_averages = pl.DataFrame([("Action", 9.0)])
    genre_averages.columns = ["genres", "weight"]

    genres = ["Action", "Horror"]

    weight = analysis.weighted_mean_for_categorical_values(genres, genre_averages)

    assert weight == 4.5


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_similarity_weight_scores_genre_list_containing_only_unseen_genres_as_zero():
    genre_averages = pl.DataFrame([("Romance", 9.0)])
    genre_averages.columns = ["genres", "weight"]

    original = ["Action", "Horror"]

    weight = analysis.weighted_mean_for_categorical_values(original, genre_averages)

    assert weight == 0.0


def test_categorical_uses_index_if_given():
    original1 = pl.Series([[1, 2, 3], [4, 5, 6]], index=[4, 5])

    original2 = pl.Series([[2, 3, 4], [1, 2, 3]], index=[1, 2])

    similarity = analysis.categorical_similarity(original1, original2, original2.index)

    assert similarity.index is not None
    assert similarity.index.to_list() == original2.index.to_list()


def test_get_mean_uses_default():
    df = pl.DataFrame({"score": [None, None, None]})

    assert analysis.get_mean_score(df, 5) == 5
