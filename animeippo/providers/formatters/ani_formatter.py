import pandas as pd
import datetime
import fast_json_normalize

from . import util
from animeippo.providers.formatters.schema import DefaultMapper, SingleMapper, MultiMapper, Columns


def transform_seasonal_data(data, feature_names):
    original = fast_json_normalize.fast_json_normalize(data["data"]["media"])

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
        Columns.DIRECTOR,
    ]

    return util.transform_to_animeippo_format(original, feature_names, keys, ANILIST_MAPPING)


def transform_watchlist_data(data, feature_names):
    original = fast_json_normalize.fast_json_normalize(data["data"])

    original.columns = [x.removeprefix("media.") for x in original.columns]

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
        Columns.DIRECTOR,
    ]

    return util.transform_to_animeippo_format(original, feature_names, keys, ANILIST_MAPPING)


def transform_user_manga_list_data(data, feature_names):
    original = fast_json_normalize.fast_json_normalize(data["data"])

    original.columns = [x.removeprefix("media.") for x in original.columns]

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


def get_ranks(tags, genres):
    ranks = {}

    for tag in tags:
        value = tag["rank"]
        ranks[tag["name"]] = value / 100 if value is not None else 0

    for genre in genres:
        ranks[genre] = 1

    return ranks


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


def get_staff(staffs, nodes, role):
    roles = [edge["role"] for edge in staffs]
    staff_ids = [node["id"] for node in nodes]

    return [staff_ids[i] for i, r in enumerate(roles) if r == role]


# fmt: off

ANILIST_MAPPING = {
    Columns.ID:                 DefaultMapper("id"),
    Columns.ID_MAL:             DefaultMapper("idMal"),
    Columns.TITLE:              DefaultMapper("title.romaji"),
    Columns.FORMAT:             DefaultMapper("format"),
    Columns.GENRES:             DefaultMapper("genres"),
    Columns.COVER_IMAGE:        DefaultMapper("coverImage.large"),
    Columns.MEAN_SCORE:         DefaultMapper("meanScore"),
    Columns.POPULARITY:         DefaultMapper("popularity"),
    Columns.DURATION:           DefaultMapper("duration"),
    Columns.EPISODES:           DefaultMapper("episodes"),
    Columns.USER_STATUS:        SingleMapper("status", str.lower),
    Columns.STATUS:             SingleMapper("status", str.lower),
    Columns.SCORE:              SingleMapper("score", util.get_score),
    Columns.SOURCE:             SingleMapper("source", 
                                             lambda source: source.lower() 
                                             if source else pd.NA),
    Columns.TAGS:               SingleMapper("tags", get_tags),
    Columns.CONTINUATION_TO:    SingleMapper("relations.edges", get_continuation),
    Columns.ADAPTATION_OF:      SingleMapper("relations.edges", get_adaptation),
    Columns.NSFW_TAGS:          SingleMapper("tags", get_nsfw_tags),
    Columns.STUDIOS:            SingleMapper("studios.edges", get_studios),
    Columns.RANKS:              MultiMapper(
                                    lambda row: get_ranks(
                                        row["tags"],
                                        row["genres"],
                                    )
                                ),
    Columns.DIRECTOR:           MultiMapper(
                                    lambda row: get_staff(
                                        row["staff.edges"],
                                        row["staff.nodes"],
                                        "Director"
                                    )
                                ),
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
                                        row["seasonYear"], 
                                        row["season"]
                                    ),
                                ),
}
# fmt: on
