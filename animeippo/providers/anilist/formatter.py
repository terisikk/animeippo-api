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


def transform_seasonal_data(data, feature_names):
    # Believe me, with polars 0.20 this is way faster than
    # pl.DataFrame(fast_json_normalize(data["data"]["media"], to_pandas=False))
    original = pl.from_pandas(fast_json_normalize(data["data"]["media"]))

    return util.transform_to_animeippo_format(
        original, feature_names, ANI_SEASONAL_SCHEMA, ANILIST_MAPPING
    )


def transform_watchlist_data(data, feature_names):
    original = fast_json_normalize(data["data"])
    original.columns = [x.removeprefix("media.") for x in original.columns]

    original = pl.from_pandas(original)

    return util.transform_to_animeippo_format(
        original, feature_names, ANI_WATCHLIST_SCHEMA, ANILIST_MAPPING
    )


def transform_user_manga_list_data(data, feature_names):
    original = fast_json_normalize(data["data"])
    original.columns = [x.removeprefix("media.") for x in original.columns]

    original = pl.from_pandas(original)

    return util.transform_to_animeippo_format(
        original, feature_names, ANI_MANGA_SCHEMA, ANILIST_MAPPING
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


def get_tags(dataframe):
    return dataframe.select(pl.col("tags").list.eval(pl.element().struct.field("name"))).to_series()


def get_user_complete_date(dataframe):
    return dataframe.select(
        pl.date(pl.col("completedAt.year"), pl.col("completedAt.month"), pl.col("completedAt.day"))
    ).to_series()


def get_studios(dataframe):
    return dataframe.select(
        pl.col("studios.edges")
        .list.eval(
            pl.when(pl.element().struct.field("node").struct.field("isAnimationStudio")).then(
                pl.element().struct.field("node").struct.field("name")
            )
        )
        .list.drop_nulls()
    ).to_series()


def get_staff(dataframe):
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
                                    .cast(
                                        pl.Enum(["winter", "spring", "summer", "fall"])
                                    )
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
    Columns.TAGS:               QueryMapper(get_tags),
    Columns.CONTINUATION_TO:    QueryMapper(get_continuation),
    Columns.ADAPTATION_OF:      QueryMapper(get_adaptation),
    Columns.STUDIOS:            QueryMapper(get_studios),
    Columns.USER_COMPLETE_DATE: QueryMapper(get_user_complete_date),
    Columns.TEMP_RANKS:         DefaultMapper("tags"),
    # Columns.DIRECTOR:           MultiMapper(["staff.edges", "staff.nodes"], 
    #                                        functools.partial(get_staff, role="Director")),
    Columns.DIRECTOR:           QueryMapper(get_staff),
}
# fmt: on
