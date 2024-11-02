import polars as pl

from animeippo.providers.columns import Columns

MIXED_MAL_WATCHLIST_SCHEMA = {
    Columns.ID: pl.UInt32,
    Columns.USER_STATUS: pl.Utf8,
    Columns.SCORE: pl.UInt8,
    Columns.USER_COMPLETE_DATE: pl.Date,
}

MIXED_ANI_WATCHLIST_SCHEMA = {
    Columns.ID: pl.UInt32,
    Columns.ID_MAL: pl.UInt32,
    Columns.TITLE: pl.Utf8,
    Columns.FORMAT: pl.Utf8,
    Columns.GENRES: pl.List(pl.Utf8),
    Columns.TAGS: pl.List(pl.Utf8),
    Columns.COVER_IMAGE: pl.Utf8,
    Columns.MEAN_SCORE: pl.Float32,
    Columns.DURATION: pl.UInt32,
    Columns.EPISODES: pl.UInt16,
    Columns.SOURCE: pl.Utf8,
    Columns.TEMP_RANKS: pl.List(
        pl.Struct({"isAdult": pl.Boolean, "name": pl.Utf8, "category": pl.Utf8, "rank": pl.UInt8})
    ),
    Columns.STUDIOS: pl.List(pl.Utf8),
    Columns.SEASON_YEAR: pl.UInt16,
    Columns.SEASON: pl.Utf8,
    Columns.DIRECTOR: pl.List(pl.UInt32),
}

MIXED_ANI_SEASONAL_SCHEMA = {
    Columns.ID: pl.UInt32,
    Columns.ID_MAL: pl.UInt32,
    Columns.TITLE: pl.Utf8,
    Columns.FORMAT: pl.Utf8,
    Columns.STATUS: pl.Utf8,
    Columns.GENRES: pl.List(pl.Utf8),
    Columns.TAGS: pl.List(pl.Utf8),
    Columns.COVER_IMAGE: pl.Utf8,
    Columns.MEAN_SCORE: pl.Float32,
    Columns.POPULARITY: pl.UInt32,
    Columns.DURATION: pl.UInt32,
    Columns.EPISODES: pl.UInt16,
    Columns.SOURCE: pl.Utf8,
    Columns.CONTINUATION_TO: pl.List(pl.UInt32),
    Columns.ADAPTATION_OF: pl.List(pl.UInt32),
    Columns.TEMP_RANKS: pl.List(
        pl.Struct({"isAdult": pl.Boolean, "name": pl.Utf8, "category": pl.Utf8, "rank": pl.UInt8})
    ),
    Columns.STUDIOS: pl.List(pl.Utf8),
    Columns.SEASON_YEAR: pl.UInt16,
    Columns.SEASON: pl.Utf8,
    Columns.DIRECTOR: pl.List(pl.UInt32),
    Columns.USER_COMPLETE_DATE: pl.Date,
}
