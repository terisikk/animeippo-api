from datetime import datetime

import numpy as np
import polars as pl
from fast_json_normalize import fast_json_normalize

from animeippo.providers.columns import (
    Columns,
)
from animeippo.providers.mappers import DefaultMapper, MultiMapper, SelectorMapper, SingleMapper
from animeippo.providers.myanimelist.schema import (
    MAL_WATCHLIST_SCHEMA,
)

from .. import util


def transform_watchlist_data(data, feature_names):
    original = pl.from_pandas(fast_json_normalize(data["data"]))

    return util.transform_to_animeippo_format(
        original, feature_names, MAL_WATCHLIST_SCHEMA, MAL_MAPPING
    )


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
        "currently_airing": "RELEASING",
        "finished_airing": "FINISHED",
        "not_yet_aired": "NOT_YET_RELEASED",
        "finished": "FINISHED",
        "currently_publishing": "RELEASING",
        "not_yet_published": "NOT_YET_RELEASED",
    }

    return mapping.get(status, status)


# fmt: off
MAL_MAPPING = {
    Columns.ID:                 DefaultMapper("node.id"),
    Columns.TITLE:              DefaultMapper("node.title"),
    Columns.FORMAT:             SelectorMapper(pl.col("node.media_type").str.to_uppercase()),
    Columns.COVER_IMAGE:        DefaultMapper("node.main_picture.medium"),
    Columns.MEAN_SCORE:         DefaultMapper("node.mean"),
    Columns.POPULARITY:         DefaultMapper("node.num_list_users"),
    Columns.DURATION:           DefaultMapper("node.average_episode_duration"),
    Columns.EPISODES:           DefaultMapper("node.num_episodes"),
    Columns.RATING:             DefaultMapper("node.rating"),
    Columns.SOURCE:             SelectorMapper(
                                    pl.col("node.source").str.to_uppercase()
                                ),
    Columns.SEASON:             SelectorMapper(
                                    pl.col("node.start_season.season").str.to_uppercase()
                                ),
    Columns.SEASON_YEAR:        DefaultMapper("node.start_season.year"),
    Columns.USER_STATUS:        SelectorMapper(
                                    pl.col("list_status.status").replace(
                                        {"watching": "CURRENT", "reading": "CURRENT",
                                         "on_hold": "PAUSED",
                                         "plan_to_watch": "PLANNING",
                                         "plan_to_read": "PLANNING",
                                         "completed": "COMPLETED", "dropped": "DROPPED"},
                                    )
                                ),
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
