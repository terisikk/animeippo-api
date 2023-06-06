import numpy as np

from animeippo.providers.formatters import util


def test_get_features_works_for_different_data_types():
    features = util.get_features(
        {"features": [1, 2, 3], "features2": "test"}, ["features", "features2"]
    )

    assert features == [1, 2, 3, "test"]


def test_get_features_ignores_null_values():
    features = util.get_features(
        {"features": [1, 2, 3, None], "features2": np.nan}, ["features", "features2"]
    )

    assert features == [1, 2, 3]


def test_features_is_none_if_no_feature_names():
    features = util.get_features({"features": [1, 2, 3, None], "features2": "test"}, None)

    assert features == []
