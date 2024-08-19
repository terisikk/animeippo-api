import numpy as np
import polars as pl

from animeippo.analysis import statistics


class EncoderStub:
    def encode(self, values):
        return [np.array(value) for value in values]


def test_genre_average_scores():
    original = pl.DataFrame(
        {
            "genres": [["Action"], ["Action", "Horror"], ["Action", "Horror", "Romance"]],
            "score": [10, 10, 7],
        }
    )

    avg = statistics.mean_score_per_categorical(original.explode("genres"), "genres")

    assert avg.sort("genres")["score"].to_list() == [9.0, 8.5, 7.0]


def test_weighted_mean_for_categoricals():
    genre_averages = pl.DataFrame([("Action", 9.0), ("Horror", 8.0), ("Romance", 7.0)])
    genre_averages.columns = ["name", "weight"]

    original = pl.DataFrame(
        {
            "id": [1, 2],
            "genres": [["Action", "Horror"], ["Action", "Romance"]],
        }
    )

    weights = statistics.weighted_mean_for_categorical_values(
        original.explode("genres"), "genres", genre_averages
    )

    assert weights.sort(descending=True).to_list() == [8.5, 8.0]


def test_weighted_mean_uses_zero_to_subsitute_nan():
    genre_averages = pl.DataFrame([("Action", 9.0)])
    genre_averages.columns = ["name", "weight"]

    original = pl.DataFrame(
        {
            "id": [1],
            "genres": [["Action", "Horror"]],
        }
    )

    weights = statistics.weighted_mean_for_categorical_values(
        original.explode("genres"), "genres", genre_averages
    )

    assert weights.to_list() == [4.5]


def test_weighted_mean_scores_genre_list_containing_only_unseen_genres_as_zero():
    genre_averages = pl.DataFrame([("Romance", 9.0)])
    genre_averages.columns = ["name", "weight"]

    original = pl.DataFrame(
        {
            "id": [1],
            "genres": [["Action", "Horror"]],
        }
    )

    weights = statistics.weighted_mean_for_categorical_values(
        original.explode("genres"), "genres", genre_averages
    )

    assert weights.to_list() == [0.0]


def test_weighted_functions_return_default_if_no_weights():
    assert statistics.weighted_mean_for_categorical_values(None, None, None) == 0.0
    assert statistics.weighted_sum_for_categorical_values(None, None, None) == 0.0

    assert (
        len(
            statistics.weight_categoricals_correlation(
                pl.DataFrame({"test": [], "score": []}), "test"
            )
        )
        == 0
    )


def test_get_mean_uses_default():
    df = pl.DataFrame({"score": [None, None, None]})

    assert statistics.mean_score_default(df, 5) == 5
