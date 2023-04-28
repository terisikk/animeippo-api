import animeippo.recommendation.analysis as analysis
import pandas as pd
import pytest


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
    original = pd.DataFrame(
        {
            "genres": [["Action"], ["Action", "Horror"], ["Action", "Horror", "Romance"]],
            "score": [10, 10, 7],
        }
    )

    avg = analysis.mean_score_per_categorical(original, "genres")

    assert avg.tolist() == [9.0, 8.5, 7.0]


def test_similarity_weights():
    genre_averages = pd.Series(data=[9.0, 8.0, 7.0], index=["Action", "Horror", "Romance"])

    original = pd.DataFrame(
        {"title": ["Hellsing", "Inuyasha"], "genres": [["Action", "Horror"], ["Action", "Romance"]]}
    )

    weights = original["genres"].apply(
        analysis.weighted_mean_for_categorical_values, args=(genre_averages,)
    )

    assert weights.tolist() == [8.5, 8.0]


def test_similarity_weight_uses_zero_to_subsitute_nan():
    genre_averages = pd.Series(data=[9.0], index=["Action"])

    genres = ["Action", "Horror"]

    weight = analysis.weighted_mean_for_categorical_values(genres, genre_averages)

    assert weight == 4.5


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_similarity_weight_scores_genre_list_containing_only_unseen_genres_as_zero():
    genre_averages = pd.Series(data=[9.0], index=["Romance"])

    original = ["Action", "Horror"]

    weight = analysis.weighted_mean_for_categorical_values(original, genre_averages)

    assert weight == 0.0


def test_weight_categoricals_z_score():
    original = pd.DataFrame(
        {
            "genres": [["Action"], ["Action", "Horror"], ["Action", "Horror", "Romance"]],
            "score": [10, 10, 7],
        }
    )

    scores = analysis.weight_categoricals_z_score(original, "genres")

    assert scores.at["Action", 0] > scores.at["Horror", 0] > scores.at["Romance", 0]
