from animeippo.providers.columns import Columns


import polars as pl


ANI_WATCHLIST_SCHEMA = {
    Columns.ID: pl.Int64,
    Columns.ID_MAL: pl.Int64,
    Columns.TITLE: pl.Utf8,
    Columns.FORMAT: pl.Utf8,
    Columns.GENRES: pl.List(pl.Utf8),
    Columns.TAGS: pl.List(pl.Utf8),
    Columns.COVER_IMAGE: pl.Utf8,
    Columns.SCORE: pl.Int64,
    Columns.MEAN_SCORE: pl.Float64,
    Columns.DURATION: pl.Int64,
    Columns.EPISODES: pl.Int64,
    Columns.SOURCE: pl.Utf8,
    Columns.TEMP_RANKS: pl.Struct,
    Columns.STUDIOS: pl.List,
    Columns.SEASON_YEAR: pl.Int64,
    Columns.SEASON: pl.Utf8,
    Columns.DIRECTOR: pl.List(pl.Int64),
    Columns.USER_STATUS: pl.Utf8,
    Columns.USER_COMPLETE_DATE: pl.Date,
}
ANI_SEASONAL_SCHEMA = {
    Columns.ID: pl.Int64,
    Columns.ID_MAL: pl.Int64,
    Columns.TITLE: pl.Utf8,
    Columns.FORMAT: pl.Utf8,
    Columns.STATUS: pl.Utf8,
    Columns.GENRES: pl.List(pl.Utf8),
    Columns.TAGS: pl.List(pl.Utf8),
    Columns.COVER_IMAGE: pl.Utf8,
    Columns.MEAN_SCORE: pl.Float64,
    Columns.POPULARITY: pl.Int64,
    Columns.DURATION: pl.Int64,
    Columns.EPISODES: pl.Int64,
    Columns.SOURCE: pl.Utf8,
    Columns.CONTINUATION_TO: pl.List(pl.Int64),
    Columns.ADAPTATION_OF: pl.List(pl.Int64),
    Columns.TEMP_RANKS: pl.Struct,
    Columns.STUDIOS: pl.List,
    Columns.SEASON_YEAR: pl.Int64,
    Columns.SEASON: pl.Utf8,
    Columns.DIRECTOR: pl.List,
    Columns.USER_COMPLETE_DATE: pl.Date,
}
ANI_MANGA_SCHEMA = {
    Columns.ID: pl.Int64,
    Columns.ID_MAL: pl.Int64,
    Columns.TITLE: pl.Utf8,
    Columns.GENRES: pl.List(pl.Utf8),
    Columns.TAGS: pl.List(pl.Utf8),
    Columns.SCORE: pl.Int64,
    Columns.MEAN_SCORE: pl.Float64,
    Columns.STATUS: pl.Utf8,
    Columns.USER_STATUS: pl.Utf8,
    Columns.USER_COMPLETE_DATE: pl.Date,
}
