import numpy as np
import polars as pl

import animeippo.analysis.statistics


def test_calculate_residuals():
    expected = [32]
    actual = animeippo.analysis.statistics.calculate_residuals(np.array([20]), np.array([10]))
    assert np.rint(actual) == np.rint(expected)


def test_extract_features():
    df = pl.DataFrame(
        {
            "genres": [
                ["Action", "Drama", "Horror"],
                ["Action", "Shounen", "Romance"],
                ["Action", "Historical", "Comedy"],
                ["Shounen", "Drama"],
                ["Drama", "Historical"],
            ],
            "cluster": [0, 0, 0, 1, 2],
        }
    )

    gdf = df.explode("genres")

    features = animeippo.analysis.statistics.get_descriptive_features(
        gdf, "genres", "cluster", 2, min_count=1
    )

    assert features.select(pl.exclude("cluster")).rows() == [
        ("Action", "Comedy"),
        ("Shounen", "Drama"),
        ("Historical", "Drama"),
    ]


def test_extract_features_without_feature_count():
    """Test TF-IDF-based feature extraction returns all features ranked by distinctiveness."""
    df = pl.DataFrame(
        {
            "genres": [
                ["Action", "Drama", "Horror"],
                ["Action", "Shounen", "Romance"],
                ["Action", "Historical", "Comedy"],
                ["Shounen", "Drama"],
                ["Drama", "Historical"],
            ],
            "cluster": [0, 0, 0, 1, 2],
        }
    )

    gdf = df.explode("genres")

    features = animeippo.analysis.statistics.get_descriptive_features(
        gdf, "genres", "cluster", min_count=1
    )

    # TF-IDF ranks features by distinctiveness (rare across clusters but common within)
    # Cluster 0: Action (in all 3 items), Comedy/Horror/Romance (distinctive)
    # Cluster 1: Shounen (only in this cluster), Drama (shared)
    # Cluster 2: Historical (mostly in this cluster), Drama (shared)
    assert features.select(pl.exclude("cluster")).rows() == [
        ("Action", "Comedy", "Horror", "Romance", "Historical", "Shounen", "Drama"),
        ("Shounen", "Drama", "Action", "Comedy", "Historical", "Horror", "Romance"),
        ("Historical", "Drama", "Action", "Comedy", "Horror", "Romance", "Shounen"),
    ]
