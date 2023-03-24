from animeippo import analysis
import pandas as pd

MAL_DATA = [
    {
        "id": 30,
        "title": "Neon Genesis Evangelion",
        "main_picture": {
            "medium": "https://api-cdn.myanimelist.net/images/anime/1314/108941.jpg",
            "large": "https://api-cdn.myanimelist.net/images/anime/1314/108941l.jpg",
        },
        "nsfw": "white",
        "genres": [
            {"id": 1, "name": "Action"},
            {"id": 5, "name": "Avant Garde"},
            {"id": 46, "name": "Award Winning"},
            {"id": 8, "name": "Drama"},
            {"id": 18, "name": "Mecha"},
            {"id": 40, "name": "Psychological"},
            {"id": 24, "name": "Sci-Fi"},
            {"id": 41, "name": "Suspense"},
        ],
        "my_list_status": {
            "status": "completed",
            "score": 10,
            "num_episodes_watched": 26,
            "is_rewatching": False,
            "updated_at": "2013-08-09T06:11:09+00:00",
            "tags": [],
        },
    },
    {
        "id": 270,
        "title": "Hellsing",
        "main_picture": {
            "medium": "https://api-cdn.myanimelist.net/images/anime/10/19956.jpg",
            "large": "https://api-cdn.myanimelist.net/images/anime/10/19956l.jpg",
        },
        "nsfw": "white",
        "genres": [
            {"id": 1, "name": "Action"},
            {"id": 50, "name": "Adult Cast"},
            {"id": 58, "name": "Gore"},
            {"id": 14, "name": "Horror"},
            {"id": 42, "name": "Seinen"},
            {"id": 37, "name": "Supernatural"},
            {"id": 32, "name": "Vampire"},
        ],
        "my_list_status": {
            "status": "completed",
            "score": 8,
            "num_episodes_watched": 13,
            "is_rewatching": False,
            "updated_at": "2017-05-26T18:32:10+00:00",
            "tags": [],
        },
    },
]


def test_dataframe_can_be_constructed_from_mal():
    analysis.NCLUSTERS = 2
    data, _ = analysis.transform_mal_data(MAL_DATA)

    assert type(data) == pd.DataFrame
    assert len(data) == 2
    assert data.loc[1, "title"] == "Hellsing"
    assert data.loc[1, "cluster"] == 0


def test_mal_genres_can_be_split():
    original = [
        {"id": 1, "name": "Action"},
        {"id": 50, "name": "Adult Cast"},
        {"id": 58, "name": "Gore"},
        {"id": 14, "name": "Horror"},
        {"id": 42, "name": "Seinen"},
        {"id": 37, "name": "Supernatural"},
        {"id": 32, "name": "Vampire"},
    ]

    actual = analysis.split_mal_genres(original)

    expected = ["Action", "Adult Cast", "Gore", "Horror", "Seinen", "Supernatural", "Vampire"]

    assert actual == expected


def test_genre_clustering():
    df = pd.DataFrame({"genres": [["Action", "Drama", "Horror"], ["Action", "Shounen", "Romance"]]})
    actual = analysis.get_genre_clustering(df, 2)
    expected = [1, 0]

    assert list(actual) == list(expected)


def test_one_hot_genre():
    df = pd.DataFrame({"genres": [["Action", "Drama", "Horror"], ["Action", "Shounen", "Romance"]]})
    actual = analysis.one_hot_genres(df["genres"])

    expected_0 = [1, 1, 1, 0, 0]
    expected_1 = [1, 0, 0, 1, 1]

    assert list(actual[0]) == expected_0
    assert list(actual[1]) == expected_1


def test_describe_dataframe():
    df = pd.DataFrame(
        {
            "genres": [
                ["Action", "Drama", "Horror"],
                ["Action", "Shounen", "Romance"],
                ["Action", "Historical", "Comedy"],
                ["Shounen", "Drama"],
                ["Drama", "Historical"],
            ]
        }
    )

    actual = analysis.describe_dataframe(df)
    expected = ["Action", "Drama"]

    assert actual == expected


def test_sort_genres_by_count():
    df = pd.DataFrame(
        {
            "genres": [
                ["Action", "Drama", "Horror"],
                ["Action", "Shounen", "Romance"],
                ["Action", "Historical", "Comedy"],
                ["Shounen", "Drama"],
                ["Drama", "Historical"],
            ]
        }
    )

    actual = analysis.sort_genres_by_count(df["genres"])

    assert actual.head(1).index[0] == "Action"
    assert len(actual) == 7


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
