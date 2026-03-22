import polars as pl
import pytest

from animeippo.clustering import model


def test_clustering():
    ml = model.AnimeClustering(distance_threshold=0.33)

    series = pl.DataFrame({"encoded": [{"a": 0, "b": 1, "c": 2}, {"a": 1, "b": 2, "c": 3}]})
    clusters = ml.cluster_by_features(series)

    assert clusters.tolist() == [1, 0]


def test_clustering_with_cosine():
    ml = model.AnimeClustering(distance_metric="cosine")

    series = pl.DataFrame(
        {"encoded": [{"a": 0, "b": 0, "c": 0}, {"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2, "c": 3}]}
    )
    clusters = ml.cluster_by_features(series)

    # Zero-vectors get cluster -1
    assert clusters.tolist() == [-1, 0, 0]


def test_predict_cannot_be_called_before_clustering():
    ml = model.AnimeClustering()

    with pytest.raises(RuntimeError):
        ml.predict(pl.Series([{"a": 0, "b": 1, "c": 2}]))


def test_clustering_with_franchise_reduces_distance():
    ml = model.AnimeClustering(
        distance_metric="cosine", distance_threshold=0.5, franchise_reduction=True
    )

    # Items 0 and 1 share a franchise (via relations) and are somewhat similar
    # Item 2 is different
    series = pl.DataFrame(
        {
            "id": [100, 200, 300],
            "encoded": [
                {"a": 10, "b": 5, "c": 0, "d": 0},
                {"a": 8, "b": 0, "c": 5, "d": 0},
                {"a": 0, "b": 0, "c": 10, "d": 10},
            ],
            "franchise": [["franchise_1"], ["franchise_1"], []],
            "franchise_relations": [
                [{"related_id": 200, "relation_type": "SEQUEL"}],
                [{"related_id": 100, "relation_type": "PREQUEL"}],
                [],
            ],
        }
    )

    clusters = ml.cluster_by_features(series)

    # Items 0 and 1 should be in the same cluster due to franchise reduction
    assert clusters[0] == clusters[1]
    assert clusters[0] != clusters[2]


def test_franchise_reduction_respects_distance_factor():
    ml = model.AnimeClustering(
        distance_metric="cosine", distance_threshold=0.3, franchise_reduction=True
    )

    # Items 0 and 1 share franchise but are very dissimilar
    series = pl.DataFrame(
        {
            "id": [100, 200, 300],
            "encoded": [
                {"a": 10, "b": 0, "c": 0},
                {"a": 0, "b": 0, "c": 10},
                {"a": 10, "b": 0, "c": 0},
            ],
            "franchise": [["franchise_1"], ["franchise_1"], []],
            "franchise_relations": [
                [{"related_id": 200, "relation_type": "SIDE_STORY"}],
                [{"related_id": 100, "relation_type": "PARENT"}],
                [],
            ],
        }
    )

    clusters = ml.cluster_by_features(series)

    # Items 0 and 1 are orthogonal (cosine distance = 1.0)
    # Even with related reduction (1.0 * 0.6 = 0.6), still above threshold 0.3
    assert clusters[0] != clusters[1]


def test_franchise_reduction_skips_zero_vector_members():
    ml = model.AnimeClustering(distance_metric="cosine", franchise_reduction=True)

    series = pl.DataFrame(
        {
            "id": [100, 200, 300],
            "encoded": [
                {"a": 10, "b": 5},
                {"a": 0, "b": 0},
                {"a": 10, "b": 5},
            ],
            "franchise": [["franchise_1"], ["franchise_1"], []],
            "franchise_relations": [
                [{"related_id": 200, "relation_type": "SEQUEL"}],
                [{"related_id": 100, "relation_type": "PREQUEL"}],
                [],
            ],
        }
    )

    clusters = ml.cluster_by_features(series)

    assert clusters[1] == -1


def test_franchise_pairs_without_typed_relations():
    """Franchise column without franchise_relations still produces related pairs."""
    ml = model.AnimeClustering(distance_metric="cosine", franchise_reduction=True)

    series = pl.DataFrame(
        {
            "id": [100, 200],
            "encoded": [{"a": 10, "b": 5}, {"a": 10, "b": 5}],
            "franchise": [["franchise_1"], ["franchise_1"]],
        }
    )

    pairs = ml.get_relation_pairs(series)
    assert pairs == {(0, 1): "related"}


def test_franchise_pairs_with_external_relations():
    """External typed relations don't upgrade to 'direct', but franchise pairs still exist."""
    ml = model.AnimeClustering(distance_metric="cosine", franchise_reduction=True)

    series = pl.DataFrame(
        {
            "id": [100, 200],
            "encoded": [{"a": 10, "b": 5}, {"a": 10, "b": 5}],
            "franchise": [["franchise_1"], ["franchise_1"]],
            "franchise_relations": [
                [{"related_id": 999, "relation_type": "SEQUEL"}],
                [{"related_id": 888, "relation_type": "SEQUEL"}],
            ],
        }
    )

    pairs = ml.get_relation_pairs(series)
    # Franchise gives a "related" pair, but typed relations point outside so no "direct" upgrade
    assert pairs == {(0, 1): "related"}


def test_clustering_without_franchise_column():
    ml = model.AnimeClustering(distance_metric="cosine")

    series = pl.DataFrame({"encoded": [{"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2, "c": 3}]})
    clusters = ml.cluster_by_features(series)

    assert clusters is not None


def test_merge_small_clusters():
    ml = model.AnimeClustering(
        distance_metric="cosine", distance_threshold=0.05, min_cluster_size=3
    )

    # Two triplets + one pair that's distant enough to not cluster naturally
    series = pl.DataFrame(
        {
            "encoded": [
                {"a": 10, "b": 0, "c": 0, "d": 0, "e": 0},
                {"a": 10, "b": 1, "c": 0, "d": 0, "e": 0},
                {"a": 10, "b": 0, "c": 1, "d": 0, "e": 0},
                {"a": 0, "b": 0, "c": 0, "d": 10, "e": 0},
                {"a": 0, "b": 0, "c": 0, "d": 10, "e": 1},
                {"a": 0, "b": 0, "c": 0, "d": 10, "e": 0},
                {"a": 7, "b": 0, "c": 0, "d": 3, "e": 3},  # small pair, closer to A
                {"a": 7, "b": 0, "c": 0, "d": 3, "e": 4},
            ],
        }
    )

    clusters = ml.cluster_by_features(series)

    # Items 6,7 form a pair (below min_cluster_size=3), merged into nearest triplet
    assert clusters[0] == clusters[1] == clusters[2]  # triplet A
    assert clusters[3] == clusters[4] == clusters[5]  # triplet B
    assert clusters[6] == clusters[7]  # merged together
    assert clusters[6] == clusters[0]  # merged into triplet A (closer)


def test_predict_returns_cluster_of_the_most_similar_element():
    ml = model.AnimeClustering()

    series = pl.DataFrame(
        {
            "id": [1, 2],
            "encoded": [
                {"a": True, "b": True, "c": False, "d": False},
                {"a": False, "b": False, "c": True, "d": True},
            ],
        }
    )
    clusters = ml.cluster_by_features(series)

    actual = ml.predict(pl.Series([{"a": False, "b": False, "c": True, "d": True}]))

    assert actual[0] == clusters[1]


def test_predict_preserves_order_for_multiple_items():
    """Predict must return clusters in the same order as the input items."""
    ml = model.AnimeClustering(distance_metric="cosine", distance_threshold=0.5)

    watchlist = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "encoded": [
                {"a": 10, "b": 0, "c": 0, "d": 0},
                {"a": 10, "b": 1, "c": 0, "d": 0},
                {"a": 0, "b": 0, "c": 10, "d": 0},
                {"a": 0, "b": 0, "c": 10, "d": 1},
            ],
        }
    )
    clusters = ml.cluster_by_features(watchlist)

    cluster_a = clusters[0]
    cluster_b = clusters[2]
    assert cluster_a != cluster_b

    new_items = pl.Series(
        [
            {"a": 0, "b": 0, "c": 9, "d": 2},  # similar to cluster B
            {"a": 9, "b": 2, "c": 0, "d": 0},  # similar to cluster A
            {"a": 0, "b": 0, "c": 8, "d": 3},  # similar to cluster B
            {"a": 8, "b": 3, "c": 0, "d": 0},  # similar to cluster A
        ]
    )

    predicted = ml.predict(new_items)

    assert predicted[0] == cluster_b
    assert predicted[1] == cluster_a
    assert predicted[2] == cluster_b
    assert predicted[3] == cluster_a
