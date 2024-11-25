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

    features = animeippo.analysis.statistics.get_descriptive_features(gdf, "genres", "cluster", 2)

    assert features.select(pl.exclude("cluster")).rows() == [
        ("Action", "Comedy"),
        ("Shounen", "Drama"),
        ("Historical", "Drama"),
    ]


def test_extract_features_without_feature_count():
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

    features = animeippo.analysis.statistics.get_descriptive_features(gdf, "genres", "cluster")

    assert features.select(pl.exclude("cluster")).rows() == [
        ("Action", "Comedy", "Horror", "Romance", "Historical", "Shounen", "Drama"),
        ("Shounen", "Drama", "Comedy", "Horror", "Romance", "Historical", "Action"),
        ("Historical", "Drama", "Comedy", "Horror", "Romance", "Shounen", "Action"),
    ]
