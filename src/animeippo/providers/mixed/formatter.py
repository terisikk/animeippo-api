import polars as pl
from fast_json_normalize import fast_json_normalize

from animeippo.providers.anilist.data import GENRE_FEATURE_STRUCTS, TAG_BY_NAME
from animeippo.providers.columns import Columns
from animeippo.providers.mappers import QueryMapper, SingleMapper

from ..anilist.formatter import ANILIST_MAPPING
from ..myanimelist.formatter import MAL_MAPPING
from ..util import transform_to_animeippo_format
from .schema import (
    MIXED_ANI_MANGA_SCHEMA,
    MIXED_ANI_SEASONAL_SCHEMA,
    MIXED_ANI_WATCHLIST_SCHEMA,
    MIXED_MAL_MANGA_SCHEMA,
    MIXED_MAL_WATCHLIST_SCHEMA,
)

_TAG_MOOD_MAP = {name: info["mood"] for name, info in TAG_BY_NAME.items() if "mood" in info}
_TAG_INTENSITY_MAP = {
    name: info["intensity"] for name, info in TAG_BY_NAME.items() if "intensity" in info
}


def _enrich_mixed_features(df):
    """Build feature_info from pre-enriched tags + genres for the mixed provider."""
    tag_info = (
        df.select(pl.col("id").alias("anime_id"), "tags")
        .explode("tags")
        .filter(pl.col("tags").is_not_null())
        .with_columns(
            name=pl.col("tags").struct.field("name"),
            rank=pl.col("tags").struct.field("rank").cast(pl.UInt8),
            category=pl.col("tags").struct.field("category"),
            mood=pl.col("tags").struct.field("name").replace_strict(_TAG_MOOD_MAP, default=None),
            intensity=pl.col("tags")
            .struct.field("name")
            .replace_strict(_TAG_INTENSITY_MAP, default=None),
        )
        .select("anime_id", "name", "rank", "category", "mood", "intensity")
    )

    genre_info = (
        df.select(pl.col("id").alias("anime_id"), "genres")
        .explode("genres")
        .filter(pl.col("genres").is_not_null())
        .with_columns(info=pl.col("genres").replace_strict(GENRE_FEATURE_STRUCTS, default=None))
        .filter(pl.col("info").is_not_null())
        .unnest("info")
        .with_columns(pl.col("rank").cast(pl.UInt8))
        .select("anime_id", "name", "rank", "category", "mood", "intensity")
    )

    combined = (
        pl.concat([tag_info, genre_info])
        .group_by("anime_id", maintain_order=True)
        .agg(pl.struct(["name", "rank", "category", "mood", "intensity"]).alias("feature_info"))
    )

    return (
        df.select(pl.col("id").alias("anime_id"))
        .join(combined, on="anime_id", how="left")
        .with_columns(pl.col("feature_info").fill_null([]))
        .select("feature_info")
        .to_series()
    )


# Mixed provider gets tags pre-enriched with name/category from AniList,
# unlike the standard AniList path which gets tag IDs and enriches them
MIXED_ANI_MAPPING = {
    **ANILIST_MAPPING,
    Columns.FEATURE_INFO: QueryMapper(_enrich_mixed_features),
}


def transform_mal_watchlist_data(data):
    original = pl.from_pandas(fast_json_normalize(data["data"]))

    return transform_to_animeippo_format(original, MIXED_MAL_WATCHLIST_SCHEMA, MAL_MAPPING)


def transform_ani_watchlist_data(data, mal_df):
    original = fast_json_normalize(data["data"]["media"])
    original.columns = [x.removeprefix("media.") for x in original.columns]

    original = pl.from_pandas(original)

    df = transform_to_animeippo_format(original, MIXED_ANI_WATCHLIST_SCHEMA, MIXED_ANI_MAPPING)

    return df.join(
        mal_df.drop(Columns.FEATURES, strict=False),
        left_on=Columns.ID_MAL,
        right_on="id",
        how="left",
    )


def transform_mal_manga_data(data):
    original = pl.from_pandas(fast_json_normalize(data["data"]))

    return transform_to_animeippo_format(original, MIXED_MAL_MANGA_SCHEMA, MAL_MAPPING)


def transform_ani_manga_data(data, mal_df):
    original = fast_json_normalize(data["data"]["media"])
    original.columns = [x.removeprefix("media.") for x in original.columns]

    original = pl.from_pandas(original)

    df = transform_to_animeippo_format(original, MIXED_ANI_MANGA_SCHEMA, MIXED_ANI_MAPPING)

    return df.join(
        mal_df.drop(Columns.FEATURES, strict=False),
        left_on=Columns.ID_MAL,
        right_on="id",
        how="left",
    )


def transform_ani_seasonal_data(data):
    original = pl.from_pandas(fast_json_normalize(data["data"]["media"]))

    ani_df = transform_to_animeippo_format(original, MIXED_ANI_SEASONAL_SCHEMA, MIXED_ANI_MAPPING)

    ani_df = ani_df.with_columns(
        **{
            Columns.ADAPTATION_OF: SingleMapper("relations.edges", get_adaptation, dtype=pl.List)
            .map(original)
            .cast(pl.List(pl.UInt32)),
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
