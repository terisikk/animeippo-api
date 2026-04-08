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
        gdf, "genres", "cluster", 2, min_count=1, min_prevalence=0
    )

    result = {row["cluster"]: row["description"] for row in features.iter_rows(named=True)}

    # Top feature per cluster is deterministic; second may vary when TF-IDF scores tie
    assert result["0"][0] == "Action"
    assert result["1"][0] == "Shounen"
    assert result["2"][0] == "Historical"
    assert len(result["0"]) == 2
    assert len(result["1"]) == 2


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
        gdf, "genres", "cluster", min_count=1, min_prevalence=0
    )

    result = {row["cluster"]: row["description"] for row in features.iter_rows(named=True)}

    # TF-IDF ranks features by distinctiveness, only non-zero scores included
    assert result["0"][0] == "Action"
    assert result["1"][0] == "Shounen"
    assert result["2"][0] == "Historical"
    # Cluster 0 has more distinctive features than single-item clusters
    assert len(result["0"]) > len(result["1"])


def test_extract_features_with_prevalence_threshold():
    """Features below prevalence threshold are excluded from cluster naming."""
    df = pl.DataFrame(
        {
            "genres": [
                ["Action", "Drama"],
                ["Action", "Comedy"],
                ["Action", "Romance"],
            ],
            "cluster": [0, 0, 0],
        }
    )

    gdf = df.explode("genres")

    # With 60% threshold: only Action (3/3=100%) qualifies, Drama/Comedy/Romance (1/3=33%) don't
    features = animeippo.analysis.statistics.get_descriptive_features(
        gdf, "genres", "cluster", 2, min_count=1, min_prevalence=0.6
    )

    result = {row["cluster"]: row["description"] for row in features.iter_rows(named=True)}

    assert result["0"] == ["Action"]
