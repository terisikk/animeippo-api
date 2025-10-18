import polars as pl
from fast_json_normalize import fast_json_normalize

from animeippo.providers.anilist.schema import (
    ANI_MANGA_SCHEMA,
    ANI_SEASONAL_SCHEMA,
    ANI_WATCHLIST_SCHEMA,
)
from animeippo.providers.columns import (
    Columns,
)
from animeippo.providers.mappers import (
    DefaultMapper,
    QueryMapper,
    SelectorMapper,
)

from .. import util


def get_anilist_mapping(tag_lookup):
    """Get ANILIST_MAPPING with tag enrichment based on tag_lookup."""
    tag_lookup_df = pl.DataFrame(
        [
            {"tag_id": tag_id, "tag_name": info["name"], "tag_category": info["category"]}
            for tag_id, info in tag_lookup.items()
        ]
    )

    def enrich_tags_for_names(df):
        """Enrich tags and extract names for the tags column"""
        return (
            df.select([pl.col("id").alias("anime_id"), "tags"])
            .explode("tags")
            .unnest("tags")
            .join(tag_lookup_df, left_on="id", right_on="tag_id", how="left")
            .with_columns(
                pl.when(pl.col("tag_name").is_null())
                .then(pl.lit("Unknown"))
                .otherwise(pl.col("tag_name"))
                .alias("name")
            )
            .group_by("anime_id", maintain_order=True)
            .agg(pl.col("name"))
            .select("name")
            .to_series()
        )

    def enrich_tags_for_ranks(df):
        """Enrich tags with name, rank, and category for the temp_ranks column"""
        return (
            df.select([pl.col("id").alias("anime_id"), "tags"])
            .explode("tags")
            .unnest("tags")
            .join(tag_lookup_df, left_on="id", right_on="tag_id", how="left")
            .with_columns(
                [
                    pl.when(pl.col("tag_name").is_null())
                    .then(pl.lit("Unknown"))
                    .otherwise(pl.col("tag_name"))
                    .alias("name"),
                    pl.when(pl.col("tag_category").is_null())
                    .then(pl.lit("Theme-Other"))
                    .otherwise(pl.col("tag_category"))
                    .alias("category"),
                ]
            )
            .group_by("anime_id", maintain_order=True)
            .agg(pl.struct(["name", "rank", "category"]))
            .select("tags")
            .to_series()
        )

    ANILIST_MAPPING[Columns.TAGS] = QueryMapper(enrich_tags_for_names)
    ANILIST_MAPPING[Columns.TEMP_RANKS] = QueryMapper(enrich_tags_for_ranks)

    return ANILIST_MAPPING


def transform_seasonal_data(data, feature_names, tag_lookup):
    # Believe me, with polars 1.12 this is way faster than
    # original = pl.json_normalize(data["data"]["media"])
    original = pl.from_pandas(fast_json_normalize(data["data"]["media"]))

    return util.transform_to_animeippo_format(
        original, feature_names, ANI_SEASONAL_SCHEMA, get_anilist_mapping(tag_lookup)
    )


def transform_watchlist_data(data, feature_names, tag_lookup):
    original = fast_json_normalize(data["data"])
    original.columns = [x.removeprefix("media.") for x in original.columns]

    original = pl.from_pandas(original)

    return util.transform_to_animeippo_format(
        original, feature_names, ANI_WATCHLIST_SCHEMA, get_anilist_mapping(tag_lookup)
    )


def transform_user_manga_list_data(data, feature_names, tag_lookup):
    original = fast_json_normalize(data["data"])
    original.columns = [x.removeprefix("media.") for x in original.columns]

    original = pl.from_pandas(original)

    return util.transform_to_animeippo_format(
        original, feature_names, ANI_MANGA_SCHEMA, get_anilist_mapping(tag_lookup)
    )


def filter_relations(dataframe, meaningful_relations):
    return dataframe.select(
        pl.col("relations.edges")
        .list.eval(
            pl.when(pl.element().struct.field("relationType").is_in(meaningful_relations)).then(
                pl.element().struct.field("node").struct.field("id")
            )
        )
        .list.drop_nulls()
    ).to_series()


def get_continuation(dataframe):
    meaningful_relations = ["PARENT", "PREQUEL"]

    return filter_relations(dataframe, meaningful_relations)


def get_adaptation(field):
    meaningful_relations = ["ADAPTATION"]

    return filter_relations(field, meaningful_relations)


def get_studios():
    return (
        pl.col("studios.edges")
        .list.eval(
            pl.when(pl.element().struct.field("node").struct.field("isAnimationStudio")).then(
                pl.element().struct.field("node").struct.field("name")
            )
        )
        .list.drop_nulls()
    )


def get_staff(dataframe):
    # Could use .over() but this is 4x faster, possibly due to the overhead of explodes
    return dataframe.join(
        dataframe.select("id", "staff.edges", "staff.nodes")
        .explode(["staff.edges", "staff.nodes"])
        .select(
            "id",
            pl.when(pl.col("staff.edges").struct.field("role") == "Director").then(
                pl.col("staff.nodes").struct.field("id").alias("director")
            ),
        )
        .drop_nulls()
        .group_by(pl.col("id"))
        .agg(pl.col("director")),
        how="left",
        on="id",
    )["director"].fill_null(pl.lit([]))


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
    Columns.SEASON_YEAR:        DefaultMapper("seasonYear"),
    Columns.SEASON:             SelectorMapper(
                                    pl.col("season").str.to_lowercase()
                                ),
    Columns.USER_STATUS:        SelectorMapper(pl.col("status").str.to_lowercase()),
    Columns.STATUS:             SelectorMapper(pl.col("status").str.to_lowercase()),
    Columns.SCORE:              SelectorMapper(
                                    pl.when(pl.col("score") > 0)
                                    .then(pl.col("score"))
                                    .otherwise(None)
                                ),
    Columns.SOURCE:             SelectorMapper(
                                    pl.when(pl.col("source").is_not_null())
                                    .then(pl.col("source").str.to_lowercase())
                                    .otherwise(None)
                                ),
    Columns.TAGS:               SelectorMapper(
                                    pl.col("tags").list.eval(
                                        pl.element().struct.field("id")
                                    )
                                ),
    Columns.CONTINUATION_TO:    QueryMapper(get_continuation),
    Columns.ADAPTATION_OF:      QueryMapper(get_adaptation),
    Columns.STUDIOS:            SelectorMapper(get_studios()),
    Columns.USER_COMPLETE_DATE: SelectorMapper(
                                    pl.date(
                                        pl.col("completedAt.year"), 
                                        pl.col("completedAt.month"), 
                                        pl.col("completedAt.day")
                                    )
                                ),
    Columns.TEMP_RANKS:         DefaultMapper("tags"),
    Columns.DIRECTOR:           QueryMapper(get_staff),
}
# fmt: on
