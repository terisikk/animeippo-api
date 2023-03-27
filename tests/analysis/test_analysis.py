from animeippo.analysis import analysis
import pandas as pd


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


def test_calculate_deviation():
    expected = 2.5
    actual = analysis.calculate_deviation(20, 5, 10, 5)
    assert actual == expected


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

    cgtable = analysis.create_contingency_table(df, "genres", "cluster")

    assert cgtable.index.values.tolist() == [
        "Action",
        "Drama",
        "Horror",
        "Shounen",
        "Romance",
        "Historical",
        "Comedy",
    ]

    assert cgtable.columns.to_list() == [0, 1, 2]

    assert cgtable.loc["Action", 0] == 3
    assert cgtable.loc["Romance", 0] == 1
