import numpy as np
import pandas as pd

from datetime import datetime

from . import util

from animeippo.providers.formatters.schema import DefaultMapper, SingleMapper, MultiMapper


def transform_watchlist_data(data, feature_names):
    original = pd.json_normalize(data["data"])

    keys = [
        "id",
        "title",
        "format",
        "genres",
        "coverImage",
        "user_status",
        "mean_score",
        "rating",
        "score",
        "source",
        "studios",
        "user_complete_date",
        "start_season",
    ]

    return transform_to_animeippo_format(original, feature_names, keys)


def transform_seasonal_data(data, feature_names):
    original = pd.json_normalize(data["data"])

    keys = [
        "id",
        "title",
        "format",
        "genres",
        "coverImage",
        "mean_score",
        "popularity",
        "status",
        "score",
        "duration",
        "episodes",
        "source",
        "rating",
        "studios",
        "start_season",
    ]

    return transform_to_animeippo_format(original, feature_names, keys)


def transform_user_manga_list_data(data, feature_names):
    original = pd.json_normalize(data["data"])

    keys = [
        "id",
        "title",
        "format",
        "genres",
        "coverImage",
        "user_status",
        "mean_score",
        "rating",
        "score",
        "source",
        "studios",
        "user_complete_date",
    ]

    return transform_to_animeippo_format(original, feature_names, keys)


def transform_related_anime(data, feature_names):
    original = pd.json_normalize(data["data"])

    keys = [
        "id",
        "continuation_to",
    ]

    filtered = transform_to_animeippo_format(original, feature_names, keys)

    return filtered[~pd.isna(filtered["continuation_to"])]["continuation_to"].to_list()


def transform_to_animeippo_format(original, feature_names, keys):
    df = pd.DataFrame(columns=keys)

    if len(original) == 0:
        return df

    df = run_mappers(df, original, MAL_MAPPING)

    df["features"] = df.apply(util.get_features, args=(feature_names,), axis=1)

    if "id" in df.columns:
        df = df.drop_duplicates(subset="id")
        df = df.set_index("id")

    return df.infer_objects()


def run_mappers(dataframe, original, mapping):
    for key, mapper in mapping.items():
        if key in dataframe.columns:
            dataframe[key] = mapper.map(original)

    return dataframe


def split_id_name_field(field):
    names = []

    for item in field:
        names.append(item.get("name", np.nan))

    return names


@util.default_if_error(pd.NA)
def get_season(year, season):
    if year == 0 or pd.isna(year):
        year = "?"
    else:
        year = str(int(year))

    if pd.isna(season):
        season = "?"

    return f"{year}/{str(season).lower()}"


def filter_relations(relation, id, meaningful_relations):
    if relation in meaningful_relations and id is not None:
        return id

    return pd.NA


def get_continuation(relation, id):
    meaningful_relations = ["parent_story", "prequel"]

    return filter_relations(relation, id, meaningful_relations)


def get_image_url(field):
    return field.get("medium", pd.NA)


def get_score(score):
    # np.nan is a float, pd.NA is not, causes problems
    return score if score != 0 else np.nan


def get_user_complete_date(finish_date):
    if pd.isna(finish_date):
        return pd.NA

    return datetime.strptime(finish_date, "%Y-%m-%d")


def get_status(status):
    mapping = {
        "currently_airing": "releasing",
        "finished_airing": "finished",
        "not_yet_aired": "not_yet_released",
        "finished": "finished",
        "currently_publishing": "releasing",
        "not_yet_published": "not_yet_released",
    }

    return mapping.get(status, status)


# fmt: off
MAL_MAPPING = {
    "id": DefaultMapper("node.id"),
    "title": DefaultMapper("node.title"),
    "format": DefaultMapper("node.media_type"),
    "coverImage": DefaultMapper("node.main_picture.medium"),
    "mean_score": DefaultMapper("node.mean"),
    "popularity": DefaultMapper("node.num_list_users"),
    "duration": DefaultMapper("node.average_episode_duration"),
    "episodes": DefaultMapper("node.num_episodes"),
    "rating": DefaultMapper("node.rating"),
    "source": DefaultMapper("node.source"),
    "user_status": DefaultMapper("list_status.status"),
    "genres": SingleMapper("node.genres", split_id_name_field, []),
    "studios": SingleMapper("node.studios", split_id_name_field),
    "status": SingleMapper("node.status", get_status),
    "score": SingleMapper("list_status.score", get_score),
    # "tags": SingleMapper("media.tags", get_tags),
    # "continuation_to": SingleMapper("media.relations.edges", get_continuation),
    # "adaptation_of": SingleMapper("media.relations.edges", get_adaptation),
    # "ranks": SingleMapper("media.tags", get_ranks),
    # "nsfw_tags": SingleMapper("media.tags", get_nsfw_tags),
    "user_complete_date": SingleMapper("list_status.finish_date", get_user_complete_date),
    "start_season": MultiMapper(
        lambda row: get_season(row["node.start_season.year"], row["node.start_season.season"]),
    ),
    "continuation_to": MultiMapper(
        lambda row: get_continuation(row["relation_type"], row["node.id"])
    ),
}

# fmt: on
