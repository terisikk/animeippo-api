import ijson
import pandas as pd


def find_all_similar_anime(genres):
    genres = [genre.lower() for genre in genres]

    similar_anime = None

    with open("anime-offline-database/anime-offline-database-minified.json") as f:
        anime = ijson.items(f, "data.item")
        similar_anime = (a for a in anime if all(genre in a["tags"] for genre in genres))

    return transform_to_animeippo_format(similar_anime)


def find_by_titles(titles):
    matching_anime = None

    with open("anime-offline-database/anime-offline-database-minified.json") as f:
        anime = ijson.items(f, "data.item")
        matching_anime = (a for a in anime if a["title"] in titles)

    return transform_to_animeippo_format(matching_anime)


def transform_to_animeippo_format(data):
    df = pd.DataFrame(data)
    df.rename(columns={"tags": "genres"}, inplace=True)

    # For now, most of these will be needed later in development
    df = df.drop(
        [
            "animeSeason",
            "episodes",
            "picture",
            "relations",
            "sources",
            "status",
            "synonyms",
            "thumbnail",
        ],
        axis=1,
    )

    return df
