"""Shared cluster naming and ranking for both recommendation and analysis paths."""

import polars as pl

from animeippo.profiling.cluster_namer import ClusterNamer


def name_all_clusters(watchlist, tag_lookup, genres, nsfw_tags=None):
    """Generate names for all clusters in the watchlist.

    Returns {cluster_id_str: name} dict.
    """
    if "cluster" not in watchlist.columns or "features" not in watchlist.columns:
        return {}

    gdf = watchlist.explode("features")

    if nsfw_tags:
        gdf = gdf.filter(~pl.col("features").is_in(nsfw_tags))

    namer = ClusterNamer(tag_lookup=tag_lookup, genres=genres)
    return namer.name_clusters_from_data(gdf, "features", "cluster")


def get_cluster_stats(watchlist):
    """Compute per-cluster statistics from the watchlist."""
    if "cluster" not in watchlist.columns:
        return pl.DataFrame()

    return (
        watchlist.group_by("cluster")
        .agg(
            count=pl.col("id").count(),
            mean_score=pl.col("score").mean(),
            completed_count=(pl.col("user_status") == "COMPLETED").sum(),
        )
        .with_columns(
            completion_rate=(pl.col("completed_count") / pl.col("count") * 100).round(1),
            mean_score=pl.col("mean_score").round(1),
        )
    )


def rank_clusters(watchlist, recommendations):
    """Rank clusters by user enjoyment * content volume * match quality.

    Returns list of dicts sorted by rank_score descending.
    """
    if "cluster" not in watchlist.columns or "cluster" not in recommendations.columns:
        return []

    user_scores = (
        watchlist.filter(pl.col("cluster") >= 0)
        .group_by("cluster")
        .agg(mean_rating=pl.col("score").mean().fill_null(5.0))
    )

    rec_stats = (
        recommendations.filter(pl.col("cluster").is_not_null())
        .group_by("cluster")
        .agg(
            rec_count=pl.col("id").len(),
            mean_similarity=pl.col("cluster_similarity").mean(),
        )
    )

    return (
        user_scores.join(rec_stats, on="cluster", how="inner")
        .filter(
            pl.col("mean_similarity").is_not_null()
            & pl.col("mean_similarity").is_not_nan()
            & (pl.col("mean_similarity") > 0)
        )
        .with_columns(
            rank_score=pl.col("mean_rating")
            * pl.col("rec_count").log1p()
            * pl.col("mean_similarity")
        )
        .sort("rank_score", descending=True)
        .to_dicts()
    )
