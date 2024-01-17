import polars as pl
from fast_json_normalize import fast_json_normalize

from animeippo.providers.columns import Columns
from animeippo.providers.mappers import SingleMapper

from ..anilist.formatter import ANILIST_MAPPING
from ..myanimelist.formatter import MAL_MAPPING
from ..util import transform_to_animeippo_format


def transform_mal_watchlist_data(data, feature_names):
    original = pl.from_pandas(fast_json_normalize(data["data"]))

    keys = [
        Columns.ID,
        Columns.USER_STATUS,
        Columns.SCORE,
        Columns.USER_COMPLETE_DATE,
    ]

    return transform_to_animeippo_format(original, feature_names, keys, MAL_MAPPING)


def transform_ani_watchlist_data(data, feature_names, mal_df):
    original = fast_json_normalize(data["data"]["media"])
    original.columns = [x.removeprefix("media.") for x in original.columns]

    original = pl.from_pandas(original)

    keys = [
        Columns.ID,
        Columns.ID_MAL,
        Columns.TITLE,
        Columns.FORMAT,
        Columns.GENRES,
        Columns.COVER_IMAGE,
        Columns.MEAN_SCORE,
        Columns.SOURCE,
        Columns.TAGS,
        Columns.RANKS,
        Columns.STUDIOS,
        Columns.SEASON_YEAR,
        Columns.SEASON,
    ]

    df = transform_to_animeippo_format(original, feature_names, keys, ANILIST_MAPPING)

    return df.join(mal_df.drop(Columns.FEATURES), left_on=Columns.ID_MAL, right_on="id", how="left")


def transform_ani_seasonal_data(data, feature_names):
    original = pl.from_pandas(fast_json_normalize(data["data"]["media"]))

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
        Columns.SCORE,
        Columns.DURATION,
        Columns.EPISODES,
        Columns.SOURCE,
        Columns.TAGS,
        Columns.RANKS,
        Columns.STUDIOS,
        Columns.SEASON_YEAR,
        Columns.SEASON,
    ]

    ani_df = transform_to_animeippo_format(original, feature_names, keys, ANILIST_MAPPING)

    ani_df = ani_df.with_columns(
        **{
            Columns.ADAPTATION_OF: SingleMapper("relations.edges", get_adaptation).map(original),
        }
    )

    return ani_df


def get_adaptation(field):
    relations = []

    for item in field:
        relationType = item.get("relationType", "")
        node = item.get("node", {})
        mal_id = node.get("idMal", None)

        if relationType == "ADAPTATION" and id is not None:
            relations.append(mal_id)

    return relations
