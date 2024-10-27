import polars as pl

from animeippo.providers.columns import Columns

MAL_WATCHLIST_SCHEMA = {
    Columns.ID: pl.UInt32,
    Columns.TITLE: pl.Utf8,
    Columns.FORMAT: pl.Utf8,
    Columns.GENRES: pl.List(pl.Utf8),
    Columns.COVER_IMAGE: pl.Utf8,
    Columns.USER_STATUS: pl.Utf8,
    Columns.MEAN_SCORE: pl.Float32,
    Columns.SCORE: pl.UInt8,
    Columns.SOURCE: pl.Utf8,
    Columns.RATING: pl.Utf8,
    Columns.STUDIOS: pl.List(pl.Utf8),
    Columns.USER_COMPLETE_DATE: pl.Date,
    Columns.SEASON_YEAR: pl.UInt16,
    Columns.SEASON: pl.Utf8,
}


MAL_SEASONAL_SCHEMA = {
    Columns.ID: pl.UInt32,
    Columns.TITLE: pl.Utf8,
    Columns.FORMAT: pl.Utf8,
    Columns.GENRES: pl.List(pl.Utf8),
    Columns.COVER_IMAGE: pl.Utf8,
    Columns.MEAN_SCORE: pl.Float32,
    Columns.POPULARITY: pl.UInt32,
    Columns.STATUS: pl.Utf8,
    Columns.DURATION: pl.UInt32,
    Columns.EPISODES: pl.UInt16,
    Columns.SOURCE: pl.Utf8,
    Columns.TAGS: pl.List(pl.Utf8),
    Columns.RATING: pl.Utf8,
    Columns.STUDIOS: pl.List(pl.Utf8),
    Columns.SEASON_YEAR: pl.UInt16,
    Columns.SEASON: pl.Utf8,
}

MAL_MANGA_SCHEMA = {
    Columns.ID: pl.UInt32,
    Columns.TITLE: pl.Utf8,
    Columns.FORMAT: pl.Utf8,
    Columns.GENRES: pl.List(pl.Utf8),
    Columns.COVER_IMAGE: pl.Utf8,
    Columns.USER_STATUS: pl.Utf8,
    Columns.MEAN_SCORE: pl.Float32,
    Columns.RATING: pl.Utf8,
    Columns.SCORE: pl.UInt8,
    Columns.SOURCE: pl.Utf8,
    Columns.USER_COMPLETE_DATE: pl.Date,
}

MAL_RELATED_SCHEMA = {
    Columns.ID: pl.UInt32,
    Columns.CONTINUATION_TO: pl.List(pl.UInt32),
}
