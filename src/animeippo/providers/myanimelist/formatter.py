from datetime import datetime

import numpy as np
import polars as pl
from fast_json_normalize import fast_json_normalize

from animeippo.providers.columns import (
    Columns,
)
from animeippo.providers.mappers import DefaultMapper, MultiMapper, SelectorMapper, SingleMapper
from animeippo.providers.myanimelist.schema import (
    MAL_MANGA_SCHEMA,
    MAL_RELATED_SCHEMA,
    MAL_SEASONAL_SCHEMA,
    MAL_WATCHLIST_SCHEMA,
)

from .. import util


def combine_dataframes(dataframes):
    return pl.concat(dataframes)


def transform_watchlist_data(data, feature_names):
    original = pl.from_pandas(fast_json_normalize(data["data"]))

    return util.transform_to_animeippo_format(
        original, feature_names, MAL_WATCHLIST_SCHEMA, MAL_MAPPING
    )


def transform_seasonal_data(data, feature_names):
    original = pl.from_pandas(fast_json_normalize(data["data"]))

    return util.transform_to_animeippo_format(
        original, feature_names, MAL_SEASONAL_SCHEMA, MAL_MAPPING
    )


def transform_user_manga_list_data(data, feature_names):
    original = pl.from_pandas(fast_json_normalize(data["data"]))

    return util.transform_to_animeippo_format(
        original, feature_names, MAL_MANGA_SCHEMA, MAL_MAPPING
    )


def transform_related_anime(data, feature_names):
    original = pl.from_pandas(fast_json_normalize(data["data"]))

    filtered = util.transform_to_animeippo_format(
        original, feature_names, MAL_RELATED_SCHEMA, MAL_MAPPING
    )

    return filtered.filter(~pl.col(Columns.CONTINUATION_TO).is_null())[
        Columns.CONTINUATION_TO
    ].to_list()


def split_id_name_field(field):
    names = []

    for item in field:
        names.append(item.get("name", np.nan))

    return names


def filter_relations(relation, related_id, meaningful_relations):
    if relation in meaningful_relations and id is not None:
        return related_id

    return None


def get_continuation(relation, related_id):
    meaningful_relations = ["parent_story", "prequel"]

    return (filter_relations(relation, related_id, meaningful_relations),)


def get_image_url(field):
    return field.get("medium", None)


def get_user_complete_date(finish_date):
    if finish_date is None:
        return None

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
    Columns.ID:                 DefaultMapper("node.id"),
    Columns.TITLE:              DefaultMapper("node.title"),
    Columns.FORMAT:             DefaultMapper("node.media_type"),
    Columns.COVER_IMAGE:        DefaultMapper("node.main_picture.medium"),
    Columns.MEAN_SCORE:         DefaultMapper("node.mean"),
    Columns.POPULARITY:         DefaultMapper("node.num_list_users"),
    Columns.DURATION:           DefaultMapper("node.average_episode_duration"),
    Columns.EPISODES:           DefaultMapper("node.num_episodes"),
    Columns.RATING:             DefaultMapper("node.rating"),
    Columns.SOURCE:             DefaultMapper("node.source"),
    Columns.SEASON:             DefaultMapper("node.start_season.season"),
    Columns.SEASON_YEAR:        DefaultMapper("node.start_season.year"),
    Columns.USER_STATUS:        DefaultMapper("list_status.status"),
    Columns.GENRES:             SingleMapper("node.genres", split_id_name_field, [], pl.List),
    Columns.STUDIOS:            SingleMapper("node.studios", split_id_name_field, [], pl.List),
    Columns.STATUS:             SingleMapper("node.status", get_status),
    Columns.SCORE:              SelectorMapper(
                                    pl.when(pl.col("list_status.score") > 0)
                                    .then(pl.col("list_status.score"))
                                    .otherwise(None)
                                ),    
    Columns.USER_COMPLETE_DATE: SingleMapper(
                                    "list_status.finish_date", 
                                    get_user_complete_date,
                                    None,
                                    datetime
                                ),
    Columns.CONTINUATION_TO:    MultiMapper(["relation_type", "node.id"], get_continuation),
}

# fmt: on
