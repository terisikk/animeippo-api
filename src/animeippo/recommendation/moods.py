"""Mood and intensity classification for the recommendation funnel.

Adds metadata columns that the frontend uses for progressive filtering
and that the category system can use for mood-based categories.
"""

import polars as pl

MOOD_THRESHOLD = 100
MOOD_TAG_RANK_CUTOFF = 60

TIER_SIGNS = {"heavy": 1.0, "light": -1.0, "moderate": 0.5}

MOOD_NAMES = ["chill", "hype", "emotional", "dark", "funny", "cerebral", "adventurous", "sporty"]


def explode_features(recommendations):
    return (
        recommendations.select("id", "feature_info")
        .explode("feature_info")
        .filter(pl.col("feature_info").is_not_null())
        .unnest("feature_info")
    )


def compute_moods(features):
    """Compute moods list per show: group mood scores, threshold, collect into list."""
    return (
        features.filter((pl.col("rank") >= MOOD_TAG_RANK_CUTOFF) & pl.col("mood").is_not_null())
        .group_by("id", "mood")
        .agg(score=(pl.col("rank")).sum())
        .filter(pl.col("score") >= MOOD_THRESHOLD)
        .group_by("id")
        .agg(moods=pl.col("mood").sort())
    )


def compute_intensity(features):
    """Compute raw intensity score per show from tag/genre intensity tiers."""
    return (
        features.with_columns(
            sign=pl.col("intensity").cast(pl.Utf8).replace_strict(TIER_SIGNS, default=0.0)
        )
        .with_columns(contribution=pl.col("rank") * pl.col("sign"))
        .group_by("id")
        .agg(intensity_score=pl.col("contribution").sum().fill_null(0.0))
        .select("id", "intensity_score")
    )


def bucket_intensity(df):
    """Bucket intensity scores into thirds: light, moderate, all_in."""
    return df.with_columns(
        intensity=pl.when(pl.col("intensity_score") <= pl.col("intensity_score").quantile(1 / 3))
        .then(pl.lit("light"))
        .when(pl.col("intensity_score") >= pl.col("intensity_score").quantile(2 / 3))
        .then(pl.lit("all_in"))
        .otherwise(pl.lit("moderate"))
    ).select("id", "intensity")


def add_mood_data(recommendations):
    """Add mood and intensity columns to recommendations DataFrame."""
    features = explode_features(recommendations)

    moods_df = compute_moods(features)
    intensity_df = bucket_intensity(compute_intensity(features))

    return recommendations.join(moods_df, on="id", how="left").join(
        intensity_df, on="id", how="left"
    )
