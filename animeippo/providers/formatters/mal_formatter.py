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
        "idMal",
        "title",
        "format",
        "genres",
        "coverImage",
        "mean_score",
        "popularity",
        "status",
        "continuation_to",
        "adaptation_of",
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
        "start_season",
    ]

    return transform_to_animeippo_format(original, feature_names, keys)


def transform_related_anime(data, feature_names):
    original = pd.json_normalize(data["data"])

    keys = [
        "id",
        "title",
        "source",
        "related_anime",
    ]

    return transform_to_animeippo_format(original, feature_names, keys)


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


@util.default_if_error([])
def split_id_name_field(field):
    names = []

    for item in field:
        names.append(item.get("name", np.nan))

    return names


@util.default_if_error(pd.NA)
def split_season(season_field):
    season_ret = np.nan

    year = season_field.get("year", "?")
    season = season_field.get("season", "?")

    season_ret = f"{year}/{season}"

    return season_ret


def filter_related_anime(df):
    meaningful_relations = ["parent_story", "prequel"]
    return df[df["relation_type"].isin(meaningful_relations)]


@util.default_if_error(None)
def get_image_url(field):
    return field.get("medium", None)


def get_score(score):
    # np.nan is a float, pd.NA is not, causes problems
    return score if score != 0 else np.nan


@util.default_if_error(pd.NA)
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
    "popularity": DefaultMapper("node.popularity"),
    "duration": DefaultMapper("node.average_episode_duration"),
    "episodes": DefaultMapper("node.num_episodes"),
    "rating": DefaultMapper("node.rating"),
    "source": DefaultMapper("node.source"),
    "user_status": DefaultMapper("list_status.status"),
    "genres": SingleMapper("node.genres", split_id_name_field),
    "studios": SingleMapper("node.studios", split_id_name_field),
    "status": SingleMapper("node.status", get_status),
    "score": SingleMapper("list_status.score", get_score),
    "start_season": SingleMapper("node.start_season", split_season),
    "related_anime": SingleMapper("node.related_anime", filter_related_anime),
    # "tags": SingleMapper("media.tags", get_tags),
    # "continuation_to": SingleMapper("media.relations.edges", get_continuation),
    # "adaptation_of": SingleMapper("media.relations.edges", get_adaptation),
    # "ranks": SingleMapper("media.tags", get_ranks),
    # "nsfw_tags": SingleMapper("media.tags", get_nsfw_tags),
    "user_complete_date": SingleMapper("list_status.finish_date", get_user_complete_date),
}
# fmt: on
