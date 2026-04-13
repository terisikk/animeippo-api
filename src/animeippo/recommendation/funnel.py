"""Mood and intensity classification for the recommendation funnel.

Adds metadata columns that the frontend uses for progressive filtering
and that the category system can use for mood-based categories.
"""

import polars as pl

MOOD_THRESHOLD = 1.0
MOOD_TAG_RANK_CUTOFF = 60

TIER_SIGNS = {"heavy": 1.0, "light": -1.0, "moderate": 0.5}

MOOD_NAMES = ["chill", "hype", "emotional", "dark", "funny", "cerebral", "adventurous", "sporty"]


def _explode_features(recommendations):
    return (
        recommendations.select("id", "feature_info")
        .explode("feature_info")
        .filter(pl.col("feature_info").is_not_null())
        .unnest("feature_info")
    )


def _compute_moods(exploded, all_ids):
    """Compute moods list per show: group mood scores, threshold, collect into list."""
    mood_scores = (
        exploded.filter((pl.col("rank") >= MOOD_TAG_RANK_CUTOFF) & pl.col("mood").is_not_null())
        .group_by("id", "mood")
        .agg(score=(pl.col("rank").cast(pl.Float64) / 100.0).sum())
        .filter(pl.col("score") >= MOOD_THRESHOLD)
        .group_by("id")
        .agg(moods=pl.col("mood").sort())
    )

    return (
        all_ids.join(mood_scores, on="id", how="left")
        .with_columns(pl.col("moods").fill_null([]).cast(pl.List(pl.Utf8)))
        .select("id", "moods")
    )


def _compute_intensity(exploded, all_ids):
    """Compute raw intensity score per show from tag/genre intensity tiers."""
    scores = (
        exploded.with_columns(
            sign=pl.col("intensity").cast(pl.Utf8).replace_strict(TIER_SIGNS, default=0.0)
        )
        .with_columns(contribution=pl.col("rank").cast(pl.Float64) / 100.0 * pl.col("sign"))
        .group_by("id")
        .agg(intensity_score=pl.col("contribution").sum())
    )

    return (
        all_ids.join(scores, on="id", how="left")
        .with_columns(pl.col("intensity_score").fill_null(0.0))
        .select("id", "intensity_score")
    )


def _bucket_intensity(df):
    """Bucket intensity scores into thirds: light, moderate, all_in."""
    scores = df["intensity_score"].sort()
    n = len(scores)
    low_cutoff = scores[n // 3]
    high_cutoff = scores[2 * n // 3]

    return df.with_columns(
        intensity=pl.when(pl.col("intensity_score") <= low_cutoff)
        .then(pl.lit("light"))
        .when(pl.col("intensity_score") >= high_cutoff)
        .then(pl.lit("all_in"))
        .otherwise(pl.lit("moderate"))
    )


def add_funnel_metadata(recommendations):
    """Add mood and intensity columns to recommendations DataFrame."""
    all_ids = recommendations.select("id")
    exploded = _explode_features(recommendations)

    moods_df = _compute_moods(exploded, all_ids)
    intensity_df = _compute_intensity(exploded, all_ids)

    result = recommendations.join(moods_df, on="id").join(intensity_df, on="id")

    return _bucket_intensity(result)
