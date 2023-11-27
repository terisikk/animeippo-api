import pandas as pd

from .mal_formatter import MAL_MAPPING, transform_to_animeippo_format2 as mal_transform
from .ani_formatter import ANILIST_MAPPING, transform_to_animeippo_format as ani_transform


def transform_mal_watchlist_data(data, feature_names):
    original = pd.json_normalize(data["data"])

    keys = [
        "id",
        "user_status",
        "score",
        "user_complete_date",
    ]

    return mal_transform(original, feature_names, keys)


def transform_ani_watchlist_data(data, feature_names, mal_df):
    original = pd.json_normalize(data["data"], "media", record_prefix="media.")

    keys = [
        "id",
        "idMal",
        "title",
        "format",
        "genres",
        "coverImage",
        "mean_score",
        "source",
        "tags",
        "ranks",
        "nsfw_tags",
        "studios",
        "start_season",
    ]

    df = ani_transform(original, feature_names, keys)

    return df.join(mal_df.drop("features", axis=1), on="idMal")
