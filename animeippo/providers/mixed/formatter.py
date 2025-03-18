import polars as pl
from fast_json_normalize import fast_json_normalize

from animeippo.providers.columns import Columns
from animeippo.providers.mappers import SingleMapper

from ..anilist.formatter import ANILIST_MAPPING
from ..myanimelist.formatter import MAL_MAPPING
from ..util import transform_to_animeippo_format
from .schema import (
    MIXED_ANI_SEASONAL_SCHEMA,
    MIXED_ANI_WATCHLIST_SCHEMA,
    MIXED_MAL_WATCHLIST_SCHEMA,
)


def transform_mal_watchlist_data(data, feature_names):
    original = pl.from_pandas(fast_json_normalize(data["data"]))

    return transform_to_animeippo_format(
        original, feature_names, MIXED_MAL_WATCHLIST_SCHEMA, MAL_MAPPING
    )


def transform_ani_watchlist_data(data, feature_names, mal_df):
    original = fast_json_normalize(data["data"]["media"])
    original.columns = [x.removeprefix("media.") for x in original.columns]

    original = pl.from_pandas(original)

    df = transform_to_animeippo_format(
        original, feature_names, MIXED_ANI_WATCHLIST_SCHEMA, ANILIST_MAPPING
    )

    return df.join(
        mal_df.drop(Columns.FEATURES, strict=False),
        left_on=Columns.ID_MAL,
        right_on="id",
        how="left",
    )


def transform_ani_seasonal_data(data, feature_names):
    original = pl.from_pandas(fast_json_normalize(data["data"]["media"]))

    ani_df = transform_to_animeippo_format(
        original, feature_names, MIXED_ANI_SEASONAL_SCHEMA, ANILIST_MAPPING
    )

    ani_df = ani_df.with_columns(
        **{
            Columns.ADAPTATION_OF: SingleMapper(
                "relations.edges", get_adaptation, dtype=pl.List
            ).map(original),
        }
    )

    return ani_df


def get_adaptation(field):
    relations = []

    for item in field:
        relationType = item.get("relationType", "")
        node = item.get("node", {})
        mal_id = node.get("idMal", None)

        if relationType == "ADAPTATION" and mal_id is not None:
            relations.append(mal_id)

    return relations
