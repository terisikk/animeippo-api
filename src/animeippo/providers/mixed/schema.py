import polars as pl

from animeippo.providers.columns import (
    Columns,
    FeatureInfo,
    Format,
    MediaSource,
    MediaStatus,
    Season,
    UserStatus,
)

MIXED_MAL_WATCHLIST_SCHEMA = {
    Columns.ID: pl.UInt32,
    Columns.USER_STATUS: UserStatus,
    Columns.SCORE: pl.UInt8,
    Columns.USER_COMPLETE_DATE: pl.Date,
}

MIXED_MAL_MANGA_SCHEMA = {
    Columns.ID: pl.UInt32,
    Columns.SCORE: pl.UInt8,
    Columns.USER_STATUS: UserStatus,
}

MIXED_ANI_MANGA_SCHEMA = {
    Columns.ID: pl.UInt32,
    Columns.ID_MAL: pl.UInt32,
    Columns.TITLE: pl.Utf8,
    Columns.GENRES: pl.List(pl.Utf8),
    Columns.MEAN_SCORE: pl.Float32,
}

MIXED_ANI_WATCHLIST_SCHEMA = {
    Columns.ID: pl.UInt32,
    Columns.ID_MAL: pl.UInt32,
    Columns.TITLE: pl.Utf8,
    Columns.FORMAT: Format,
    Columns.STATUS: MediaStatus,
    Columns.GENRES: pl.List(pl.Utf8),
    Columns.COVER_IMAGE: pl.Utf8,
    Columns.MEAN_SCORE: pl.Float32,
    Columns.DURATION: pl.UInt32,
    Columns.EPISODES: pl.UInt16,
    Columns.SOURCE: MediaSource,
    Columns.FEATURE_INFO: FeatureInfo,
    Columns.STUDIOS: pl.List(pl.Utf8),
    Columns.SEASON_YEAR: pl.UInt16,
    Columns.SEASON: Season,
    Columns.FRANCHISE: pl.List(pl.Utf8),
    Columns.FRANCHISE_RELATIONS: pl.List(
        pl.Struct({"related_id": pl.UInt32, "relation_type": pl.Utf8})
    ),
}

MIXED_ANI_SEASONAL_SCHEMA = {
    Columns.ID: pl.UInt32,
    Columns.ID_MAL: pl.UInt32,
    Columns.TITLE: pl.Utf8,
    Columns.FORMAT: Format,
    Columns.STATUS: MediaStatus,
    Columns.GENRES: pl.List(pl.Utf8),
    Columns.COVER_IMAGE: pl.Utf8,
    Columns.MEAN_SCORE: pl.Float32,
    Columns.POPULARITY: pl.UInt32,
    Columns.DURATION: pl.UInt32,
    Columns.EPISODES: pl.UInt16,
    Columns.SOURCE: MediaSource,
    Columns.CONTINUATION_TO: pl.List(pl.UInt32),
    Columns.ADAPTATION_OF: pl.List(pl.UInt32),
    Columns.FEATURE_INFO: FeatureInfo,
    Columns.STUDIOS: pl.List(pl.Utf8),
    Columns.SEASON_YEAR: pl.UInt16,
    Columns.SEASON: Season,
    # Columns.DIRECTOR: pl.List(pl.UInt32),
    Columns.USER_COMPLETE_DATE: pl.Date,
    Columns.FRANCHISE_RELATIONS: pl.List(
        pl.Struct({"related_id": pl.UInt32, "relation_type": pl.Utf8})
    ),
}
