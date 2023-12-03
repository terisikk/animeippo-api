import numpy as np
import pandas as pd

from animeippo.providers.formatters import util


class StubMapper:
    def map(self, original):
        return [f"{original} ran"]


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


def test_season_can_be_extracted():
    assert util.get_season(2023, "summer") == "2023/summer"
    assert util.get_season(None, "winter") == "?/winter"
    assert util.get_season(np.nan, np.nan) == "?/?"


def test_user_score_cannot_be_zero():
    original = 0

    actual = util.get_score(original)

    assert pd.isna(actual)


def test_user_score_extraction_does_not_fail_with_invalid_data():
    util.get_score(1.0)
    util.get_score(None)
    util.get_score(np.nan)


def test_mapping_skips_keys_not_in_dataframe():
    dataframe = pd.DataFrame(columns=["test1", "test2"])
    mapping = {"test1": StubMapper(), "test3": StubMapper()}

    actual = util.run_mappers(dataframe, "test1", mapping)
    assert actual["test1"].tolist() == ["test1 ran"]
    assert "test3" not in actual.columns


def test_transformation_does_not_fail_with_empty_data():
    data = util.transform_to_animeippo_format({}, ["genres", "tags"], [], {})

    assert type(data) == pd.DataFrame
    assert len(data) == 0

    data = util.transform_to_animeippo_format(
        {"data": {"test": "test"}}, ["genres", "tags"], [], {}
    )
    assert type(data) == pd.DataFrame
    assert len(data) == 0


def test_default_if_error():
    @util.default_if_error("default")
    def test_func():
        raise TypeError("test")

    assert test_func() == "default"
