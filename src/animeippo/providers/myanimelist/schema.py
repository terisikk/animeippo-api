import polars as pl

from animeippo.providers.columns import (
    Columns,
    Format,
    MediaSource,
    Season,
    UserStatus,
)

MAL_WATCHLIST_SCHEMA = {
    Columns.ID: pl.UInt32,
    Columns.TITLE: pl.Utf8,
    Columns.FORMAT: Format,
    Columns.GENRES: pl.List(pl.Utf8),
    Columns.COVER_IMAGE: pl.Utf8,
    Columns.USER_STATUS: UserStatus,
    Columns.MEAN_SCORE: pl.Float32,
    Columns.SCORE: pl.UInt8,
    Columns.SOURCE: MediaSource,
    Columns.RATING: pl.Utf8,
    Columns.STUDIOS: pl.List(pl.Utf8),
    Columns.USER_COMPLETE_DATE: pl.Date,
    Columns.SEASON_YEAR: pl.UInt16,
    Columns.SEASON: Season,
}
