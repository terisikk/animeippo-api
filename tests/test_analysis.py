import animeippo.analysis as analysis
import pandas as pd
import numpy as np


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

    distances = analysis.jaccard_pairwise_distance(original["genres"])

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

    distances = analysis.jaccard_similarity(x_orig["genres"], y_orig["genres"])

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

    recommendations = analysis.order_by_recommendation(target_df, source_df)
    expected = "Naruto"
    actual = recommendations.iloc[0]["title"]

    assert actual == expected
    assert recommendations.columns.tolist() == ["genres", "title"]


def test_describe_clusters():
    df = pd.DataFrame(
        {
            "genres": [["Action", "Drama", "Horror"], ["Action", "Shounen", "Romance"]],
            "cluster": [0, 1],
        }
    )

    descriptions = analysis.describe_clusters(df, 2)

    assert descriptions.iloc[0].tolist() == ["Drama", "Horror"]
