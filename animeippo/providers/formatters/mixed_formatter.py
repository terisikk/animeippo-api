import pandas as pd

from .mal_formatter import MAL_MAPPING, transform_to_animeippo_format as mal_transform
from .ani_formatter import ANILIST_MAPPING, transform_to_animeippo_format as ani_transform, run_mappers
from animeippo.providers.formatters.schema import SingleMapper


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


def get_adaptation(field):
    relations = []

    for item in field:
        relationType = item.get("relationType", "")
        node = item.get("node", {})
        id = node.get("idMal", None)

        if relationType == "ADAPTATION" and id is not None:
            relations.append(id)

    return relations


def transform_ani_seasonal_data(data, feature_names):
    original = pd.json_normalize(data["data"], "media", record_prefix="media.")

    keys = [
        "id",
        "idMal",
        "title",
        "format",
        "genres",
        "coverImage",
        "mean_score",
        "popularity",
        "status",
        "continuation_to",
        "score",
        "duration",
        "episodes",
        "source",
        "tags",
        "nsfw_tags",
        "ranks",
        "studios",
        "start_season",
    ]

    ani_df = ani_transform(original, feature_names, keys)

    temp_df = pd.DataFrame()
    temp_df["adaptation_of"] = SingleMapper("media.relations.edges", get_adaptation).map(original)
    temp_df["idx"] = ani_df.index.to_list()
    temp_df = temp_df.set_index("idx")

    ani_df["adaptation_of"] = temp_df["adaptation_of"]

    return ani_df
