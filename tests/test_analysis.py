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
    data = analysis.data_from_mal(MAL_DATA)

    assert type(data) == pd.DataFrame
    assert len(data) == 2
    assert data.loc[1, "title"] == "Hellsing"


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

    expected = {"Action", "Adult Cast", "Gore", "Horror", "Seinen", "Supernatural", "Vampire"}

    assert actual == expected
