import animeippo.analysis as analysis
import pandas as pd
import numpy as np
import pytest


def test_genre_clustering():
    df = pd.DataFrame({"genres": [["Action", "Drama", "Horror"], ["Action", "Shounen", "Romance"]]})
    actual = analysis.get_genre_clustering(df, 2)
    expected = [1, 0]

    assert list(actual) == list(expected)


# TODO: Test actual encoding
def test_one_hot_genre():
    df = pd.DataFrame({"genres": [["Action", "Drama", "Horror"], ["Action", "Shounen", "Romance"]]})
    actual = analysis.one_hot_genres(df["genres"])

    assert actual[0].size == len(analysis.mal.MAL_GENRES)


def test_calculate_residuals():
    expected = np.array([10])
    actual = analysis.calculate_residuals(np.array([20]), np.array([10]))
    assert np.rint(actual) == expected


def test_jaccard_pairwise_distance():
    original = pd.DataFrame({"genres": [["Action", "Adventure"], ["Action", "Fantasy"]]})

    distances = analysis.pairwise_distance(original["genres"])

    expected0 = 0.0
    actual0 = float(distances[0][0])

    expected1 = float(2 / 3)
    actual1 = float(distances[0][1])

    assert actual0 == expected0
    assert actual1 == expected1


def test_jaccard_similarity():
    x_orig = pd.DataFrame({"genres": [["Action", "Adventure"], ["Action", "Fantasy"]]})
    y_orig = pd.DataFrame(
        {"genres": [["Action", "Adventure"], ["Action", "Fantasy"], ["Comedy", "Romance"]]}
    )

    distances = analysis.similarity(x_orig["genres"], y_orig["genres"])

    expected0 = "1.0"
    actual0 = "{:.1f}".format(distances[0][0])

    expected1 = "0.3"
    actual1 = "{:.1f}".format(distances[0][1])

    assert actual0 == expected0
    assert actual1 == expected1


def test_create_contingency_table():
    df = pd.DataFrame(
        {
            "genres": [
                ["Action", "Drama", "Horror"],
                ["Action", "Shounen", "Romance"],
                ["Action", "Historical", "Comedy"],
                ["Shounen", "Drama"],
                ["Drama", "Historical"],
            ],
            "cluster": [0, 0, 0, 1, 2],
        }
    )

    cgtable = analysis.create_genre_contingency_table(df)

    assert cgtable.index.values.tolist() == [
        "Action",
        "Comedy",
        "Drama",
        "Historical",
        "Horror",
        "Romance",
        "Shounen",
    ]

    assert cgtable.columns.to_list() == [0, 1, 2]

    assert cgtable.loc["Action", 0] == 3
    assert cgtable.loc["Romance", 0] == 1


def test_recommendation():
    source_df = pd.DataFrame(
        {
            "genres": [["Action", "Adventure"], ["Action", "Fantasy"]],
            "title": ["Bleach", "Fate/Zero"],
        }
    )
    target_df = pd.DataFrame(
        {"genres": [["Romance", "Comedy"], ["Action", "Adventure"]], "title": ["Kaguya", "Naruto"]}
    )

    recommendations = analysis.recommend_by_genre_similarity(target_df, source_df)
    expected = "Naruto"
    actual = recommendations.iloc[0]["title"]

    assert actual == expected
    assert recommendations.columns.tolist() == ["genres", "title", "recommend_score"]
    assert not recommendations["recommend_score"].isnull().values.any()


def test_weighted_recommendation():
    source_df = pd.DataFrame(
        {
            "genres": [["Action", "Adventure"], ["Fantasy", "Adventure"]],
            "title": ["Bleach", "Fate/Zero"],
            "user_score": [1, 10],
        }
    )
    target_df = pd.DataFrame(
        {
            "genres": [["Action", "Adventure"], ["Fantasy", "Adventure"]],
            "title": ["Naruto", "Inuyasha"],
        }
    )

    recommendations = analysis.recommend_by_genre_similarity(target_df, source_df, weighted=True)
    expected = "Inuyasha"
    actual = recommendations.iloc[0]["title"]

    assert actual == expected
    assert not recommendations["recommend_score"].isnull().values.any()


def test_cluster_recommendation(mocker):
    mocker.patch("animeippo.analysis.get_genre_clustering", return_value=[0, 1])

    source_df = pd.DataFrame(
        {
            "genres": [["Action", "Adventure"], ["Action", "Fantasy"]],
            "title": ["Bleach", "Fate/Zero"],
        }
    )
    target_df = pd.DataFrame(
        {"genres": [["Romance", "Comedy"], ["Action", "Adventure"]], "title": ["Kaguya", "Naruto"]}
    )

    recommendations = analysis.recommend_by_cluster(target_df, source_df)
    expected = "Naruto"
    actual = recommendations.iloc[0]["title"]

    assert actual == expected
    assert recommendations.columns.tolist() == ["genres", "title", "recommend_score"]
    assert not recommendations["recommend_score"].isnull().values.any()


def test_weighted_cluster_recommendation(mocker):
    mocker.patch("animeippo.analysis.get_genre_clustering", return_value=[0, 1])

    source_df = pd.DataFrame(
        {
            "genres": [["Action", "Adventure"], ["Fantasy", "Adventure"]],
            "title": ["Bleach", "Fate/Zero"],
            "user_score": [10, 1],
        }
    )
    target_df = pd.DataFrame(
        {
            "genres": [["Fantasy", "Adventure"], ["Action", "Adventure"]],
            "title": ["Inuyasha", "Naruto"],
        }
    )
    recommendations = analysis.recommend_by_cluster(target_df, source_df, weighted=True)
    expected = "Naruto"
    actual = recommendations.iloc[0]["title"]

    assert actual == expected
    assert recommendations.columns.tolist() == ["genres", "title", "recommend_score"]
    assert not recommendations["recommend_score"].isnull().values.any()


def test_describe_clusters():
    df = pd.DataFrame(
        {
            "genres": [["Action", "Drama", "Horror"], ["Action", "Shounen", "Romance"]],
            "cluster": [0, 1],
        }
    )

    descriptions = analysis.describe_clusters(df, 2)

    assert descriptions.iloc[0].tolist() == ["Drama", "Horror"]


def test_genre_average_scores():
    original = pd.DataFrame(
        {
            "genres": [["Action"], ["Action", "Horror"], ["Action", "Horror", "Romance"]],
            "user_score": [10, 10, 7],
        }
    )

    avg = analysis.genre_average_scores(original)

    assert avg.tolist() == [9.0, 8.5, 7.0]


def test_similarity_weights():
    genre_averages = pd.Series(data=[9.0, 8.0, 7.0], index=["Action", "Horror", "Romance"])

    original = pd.DataFrame(
        {"title": ["Hellsing", "Inuyasha"], "genres": [["Action", "Horror"], ["Action", "Romance"]]}
    )

    weights = original["genres"].apply(analysis.user_genre_weight, args=(genre_averages,))

    assert weights.tolist() == [8.5, 8.0]


def test_similarity_weight_ignores_genres_without_average():
    genre_averages = pd.Series(data=[9.0], index=["Action"])

    genres = ["Action", "Horror"]

    weight = analysis.user_genre_weight(genres, genre_averages)

    assert weight == 9.0


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_similarity_weight_scores_genre_list_containing_only_unseen_genres_as_zero():
    genre_averages = pd.Series(data=[9.0], index=["Romance"])

    original = ["Action", "Horror"]

    weight = analysis.user_genre_weight(original, genre_averages)

    assert weight == 0.0
