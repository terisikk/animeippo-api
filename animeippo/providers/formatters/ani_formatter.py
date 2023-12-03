import pandas as pd
import datetime

from . import util
from animeippo.providers.formatters.schema import DefaultMapper, SingleMapper, MultiMapper, Columns


def transform_seasonal_data(data, feature_names):
    original = pd.json_normalize(data["data"], "media", record_prefix="media.")

    keys = [
        Columns.ID,
        Columns.ID_MAL,
        Columns.TITLE,
        Columns.FORMAT,
        Columns.GENRES,
        Columns.COVER_IMAGE,
        Columns.MEAN_SCORE,
        Columns.POPULARITY,
        Columns.STATUS,
        Columns.CONTINUATION_TO,
        Columns.ADAPTATION_OF,
        Columns.SCORE,
        Columns.DURATION,
        Columns.EPISODES,
        Columns.SOURCE,
        Columns.TAGS,
        Columns.NSFW_TAGS,
        Columns.RANKS,
        Columns.STUDIOS,
        Columns.START_SEASON,
    ]

    return util.transform_to_animeippo_format(original, feature_names, keys, ANILIST_MAPPING)


def transform_watchlist_data(data, feature_names):
    original = pd.json_normalize(data["data"])

    keys = [
        Columns.ID,
        Columns.ID_MAL,
        Columns.TITLE,
        Columns.FORMAT,
        Columns.GENRES,
        Columns.COVER_IMAGE,
        Columns.USER_STATUS,
        Columns.MEAN_SCORE,
        Columns.SCORE,
        Columns.SOURCE,
        Columns.TAGS,
        Columns.RANKS,
        Columns.NSFW_TAGS,
        Columns.STUDIOS,
        Columns.USER_COMPLETE_DATE,
        Columns.START_SEASON,
    ]

    return util.transform_to_animeippo_format(original, feature_names, keys, ANILIST_MAPPING)


def transform_user_manga_list_data(data, feature_names):
    original = pd.json_normalize(data["data"])

    keys = [
        Columns.ID,
        Columns.ID_MAL,
        Columns.TITLE,
        Columns.FORMAT,
        Columns.GENRES,
        Columns.TAGS,
        Columns.USER_STATUS,
        Columns.STATUS,
        Columns.MEAN_SCORE,
        Columns.SCORE,
        Columns.USER_COMPLETE_DATE,
    ]

    return util.transform_to_animeippo_format(original, feature_names, keys, ANILIST_MAPPING)


def filter_relations(field, meaningful_relations):
    relations = []

    for item in field:
        relationType = item.get("relationType", "")
        node = item.get("node", {})
        id = node.get("id", None)

        if relationType in meaningful_relations and id is not None:
            relations.append(id)

    return relations


def get_continuation(field):
    meaningful_relations = ["PARENT", "PREQUEL"]

    return filter_relations(field, meaningful_relations)


def get_adaptation(field):
    meaningful_relations = ["ADAPTATION"]

    return filter_relations(field, meaningful_relations)


def get_tags(tags):
    return [tag["name"] for tag in tags]


def get_user_complete_date(year, month, day):
    if pd.isna(year) or pd.isna(month) or pd.isna(day):
        return pd.NA

    return datetime.date(int(year), int(month), int(day))


def get_ranks(items):
    return {item["name"]: item["rank"] / 100 for item in items}


def get_nsfw_tags(items):
    return [item["name"] for item in items if item["isAdult"] is True]


def get_studios(studios):
    return set(
        [
            studio["node"]["name"]
            for studio in studios
            if studio["node"].get("isAnimationStudio", False)
        ]
    )


# fmt: off

ANILIST_MAPPING = {
    Columns.ID:                 DefaultMapper("media.id"),
    Columns.ID_MAL:             DefaultMapper("media.idMal"),
    Columns.TITLE:              DefaultMapper("media.title.romaji"),
    Columns.FORMAT:             DefaultMapper("media.format"),
    Columns.GENRES:             DefaultMapper("media.genres"),
    Columns.COVER_IMAGE:        DefaultMapper("media.coverImage.large"),
    Columns.MEAN_SCORE:         DefaultMapper("media.meanScore"),
    Columns.POPULARITY:         DefaultMapper("media.popularity"),
    Columns.DURATION:           DefaultMapper("media.duration"),
    Columns.EPISODES:           DefaultMapper("media.episodes"),
    Columns.USER_STATUS:        SingleMapper("status", str.lower),
    Columns.STATUS:             SingleMapper("media.status", str.lower),
    Columns.SCORE:              SingleMapper("score", util.get_score),
    Columns.SOURCE:             SingleMapper("media.source", str.lower),
    Columns.TAGS:               SingleMapper("media.tags", get_tags),
    Columns.CONTINUATION_TO:    SingleMapper("media.relations.edges", get_continuation),
    Columns.ADAPTATION_OF:      SingleMapper("media.relations.edges", get_adaptation),
    Columns.RANKS:              SingleMapper("media.tags", get_ranks),
    Columns.NSFW_TAGS:          SingleMapper("media.tags", get_nsfw_tags),
    Columns.STUDIOS:            SingleMapper("media.studios.edges", get_studios),
    Columns.USER_COMPLETE_DATE: MultiMapper(
                                    lambda row: get_user_complete_date(
                                        row["completedAt.year"], 
                                        row["completedAt.month"], 
                                        row["completedAt.day"]
                                    ),
                                    pd.NaT,
                                ),
    Columns.START_SEASON:       MultiMapper(
                                    lambda row: util.get_season(
                                        row["media.seasonYear"], 
                                        row["media.season"]
                                    ),
                                ),
}
# fmt: on
