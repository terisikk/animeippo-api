import polars as pl

from animeippo.recommendation.cluster_naming import (
    get_cluster_stats,
    name_all_clusters,
    rank_clusters,
)


def test_name_all_clusters_without_cluster_column():
    watchlist = pl.DataFrame({"id": [1], "features": [["Action"]]})
    assert name_all_clusters(watchlist, {}, set()) == {}


def test_name_all_clusters_without_features_column():
    watchlist = pl.DataFrame({"id": [1], "cluster": [0]})
    assert name_all_clusters(watchlist, {}, set()) == {}


def test_get_cluster_stats_without_cluster_column():
    watchlist = pl.DataFrame({"id": [1]})
    assert len(get_cluster_stats(watchlist)) == 0


def test_get_cluster_stats():
    watchlist = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "cluster": [0, 0, 1],
            "score": [8, 9, 7],
            "user_status": ["COMPLETED", "COMPLETED", "CURRENT"],
        }
    )

    stats = get_cluster_stats(watchlist)
    assert len(stats) == 2
    assert "count" in stats.columns
    assert "mean_score" in stats.columns


def test_rank_clusters_without_cluster_column():
    watchlist = pl.DataFrame({"id": [1]})
    recs = pl.DataFrame({"id": [10]})
    assert rank_clusters(watchlist, recs) == []


def test_rank_clusters():
    watchlist = pl.DataFrame({"id": [1, 2, 3], "cluster": [0, 0, 1], "score": [9, 8, 5]})

    recs = pl.DataFrame(
        {"id": [10, 11, 12], "cluster": [0, 0, 1], "cluster_similarity": [0.8, 0.7, 0.3]}
    )

    ranked = rank_clusters(watchlist, recs)
    assert len(ranked) == 2
    # Cluster 0 should rank higher (better scores, more recs, higher similarity)
    assert ranked[0]["cluster"] == 0
