import polars as pl
from fast_json_normalize import fast_json_normalize

from animeippo.providers import util
from animeippo.providers.anilist.data import GENRE_FEATURE_STRUCTS
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


def get_anilist_mapping(tag_lookup):
    """Get ANILIST_MAPPING with tag enrichment based on tag_lookup."""

    tag_lookup_df = pl.DataFrame(
        [
            {
                "tag_id": tag_id,
                "tag_name": info["name"],
                "tag_category": info["category"],
                "tag_mood": info.get("mood"),
                "tag_intensity": info.get("intensity"),
            }
            for tag_id, info in tag_lookup.items()
        ]
    )

    def enrich_features(df):
        """Build feature_info: tags enriched with metadata + genres as structs."""
        tag_info = (
            df.select([pl.col("id").alias("anime_id"), "tags"])
            .explode("tags")
            .unnest("tags")
            .join(tag_lookup_df, left_on="id", right_on="tag_id", how="left")
            .filter(pl.col("tag_name").is_not_null())
            .select(
                "anime_id",
                pl.col("tag_name").alias("name"),
                pl.col("rank").cast(pl.UInt8),
                pl.col("tag_category").alias("category"),
                pl.col("tag_mood").alias("mood"),
                pl.col("tag_intensity").alias("intensity"),
            )
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

    ANILIST_MAPPING[Columns.FEATURE_INFO] = QueryMapper(enrich_features)

    return ANILIST_MAPPING


def transform_seasonal_data(data, tag_lookup):
    # Believe me, with polars 1.39 this is way faster than
    # original = pl.json_normalize(data["data"]["media"])
    original = pl.from_pandas(fast_json_normalize(data["data"]["media"]))

    return util.transform_to_animeippo_format(
        original, ANI_SEASONAL_SCHEMA, get_anilist_mapping(tag_lookup)
    )


def transform_watchlist_data(data, tag_lookup):
    original = fast_json_normalize(data["data"])
    # Rename entry-level "status" before stripping "media." prefix to avoid
    # collision with media.status (airing status like RELEASING/FINISHED).
    original.columns = ["userStatus" if x == "status" else x for x in original.columns]
    original.columns = [x.removeprefix("media.") for x in original.columns]

    original = pl.from_pandas(original)

    return util.transform_to_animeippo_format(
        original, ANI_WATCHLIST_SCHEMA, get_anilist_mapping(tag_lookup)
    )


def transform_user_manga_list_data(data, tag_lookup):
    original = fast_json_normalize(data["data"])
    original.columns = [x.removeprefix("media.") for x in original.columns]

    original = pl.from_pandas(original)

    return util.transform_to_animeippo_format(
        original, ANI_MANGA_SCHEMA, get_anilist_mapping(tag_lookup)
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


FRANCHISE_RELATION_TYPES = [
    "SEQUEL",
    "PREQUEL",
    "PARENT",
    "SIDE_STORY",
    "ALTERNATIVE",
    "SPIN_OFF",
    "SUMMARY",
    "COMPILATION",
]


def get_franchise_relations(dataframe):
    return filter_relations(dataframe, FRANCHISE_RELATION_TYPES)


def build_typed_franchise_relations(df):
    """Extract typed relation pairs for tiered distance reduction in clustering."""
    return df.select(
        pl.col("relations.edges")
        .list.eval(
            pl.when(pl.element().struct.field("relationType").is_in(FRANCHISE_RELATION_TYPES)).then(
                pl.struct(
                    pl.element().struct.field("node").struct.field("id").alias("related_id"),
                    pl.element().struct.field("relationType").alias("relation_type"),
                )
            )
        )
        .list.drop_nulls()
    ).to_series()


def extract_recommendations(df):
    """Extract community recommendation links as list of {recommended_id, rating} structs."""
    col_name = "recommendations.edges"
    if col_name not in df.columns:
        return pl.Series([None] * len(df))

    return df.select(
        pl.col(col_name)
        .list.eval(
            pl.when(
                pl.element().struct.field("node").struct.field("mediaRecommendation").is_not_null()
            ).then(
                pl.struct(
                    pl.element()
                    .struct.field("node")
                    .struct.field("mediaRecommendation")
                    .struct.field("id")
                    .alias("recommended_id"),
                    pl.element().struct.field("node").struct.field("rating").alias("rating"),
                )
            )
        )
        .list.drop_nulls()
    ).to_series()


def build_franchise_column(df):
    relations = get_franchise_relations(df)
    ids = df["id"]
    return util.build_franchise_ids(ids, relations)


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
    Columns.SEASON:             DefaultMapper("season"),
    Columns.USER_STATUS:        DefaultMapper("userStatus"),
    Columns.STATUS:             DefaultMapper("status"),
    Columns.SCORE:              SelectorMapper(
                                    pl.when(pl.col("score") > 0)
                                    .then(pl.col("score"))
                                    .otherwise(None)
                                ),
    Columns.SOURCE:             DefaultMapper("source"),
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
    Columns.FRANCHISE:          QueryMapper(build_franchise_column),
    Columns.FRANCHISE_RELATIONS: QueryMapper(build_typed_franchise_relations),
    Columns.RECOMMENDATIONS:     QueryMapper(extract_recommendations),
}
# fmt: on
