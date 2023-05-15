from animeippo.recommendation import dataset

import numpy as np


def test_get_features_works_for_different_data_types():
    features = dataset.get_features(
        {"features": [1, 2, 3], "features2": "test"}, ["features", "features2"]
    )

    assert features == [1, 2, 3, "test"]


def test_get_features_ignores_null_values():
    features = dataset.get_features(
        {"features": [1, 2, 3, None], "features2": np.nan}, ["features", "features2"]
    )

    assert features == [1, 2, 3]
