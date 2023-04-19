import animeippo.recommendation.util as pdutil
import pandas as pd
import numpy as np


def test_calculate_residuals():
    expected = np.array([np.sqrt(10)])
    actual = pdutil.calculate_residuals(np.array([20]), np.array([10]))
    assert np.rint(actual) == np.rint(expected)


def test_extract_features():
    df = pd.DataFrame(
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

    features = pdutil.extract_features(gdf["genres"], gdf["cluster"], 2)

    assert features.values.tolist() == [
        ["Action", "Comedy"],
        ["Shounen", "Drama"],
        ["Historical", "Drama"],
    ]
