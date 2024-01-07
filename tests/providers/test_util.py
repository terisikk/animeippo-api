import numpy as np
import polars as pl

from animeippo.providers.formatters import util


class StubMapper:
    def map(self, original):
        return [f"{original} ran"]


def test_get_features():
    original = pl.DataFrame({"features1": ["1", "2", "3"], "features2": ["test", "test", "test"]})

    features = util.get_features(original, ["features1", "features2"])

    assert features[0].to_list() == ["1", "test"]


def test_user_score_cannot_be_zero():
    original = 0

    actual = util.get_score(original)

    assert actual is None


def test_user_score_extraction_does_not_fail_with_invalid_data():
    util.get_score(1.0)
    util.get_score(None)
    util.get_score(np.nan)


def test_mapping_skips_keys_not_in_dataframe():
    dataframe = pl.DataFrame({"test1": [1], "test2": [2]})
    mapping = {"test1": StubMapper(), "test3": StubMapper()}

    actual = util.run_mappers(dataframe, "test1", mapping)
    assert actual["test1"].to_list() == [["test1 ran"]]
    assert "test3" not in actual.columns


def test_transformation_does_not_fail_with_empty_data():
    data = util.transform_to_animeippo_format(pl.DataFrame(), ["genres", "tags"], [], {})

    assert type(data) == pl.DataFrame
    assert len(data) == 0

    data = util.transform_to_animeippo_format(
        pl.DataFrame({"data": {"test": "test"}}), ["genres", "tags"], [], {}
    )
    assert type(data) == pl.DataFrame
    assert len(data) == 0
