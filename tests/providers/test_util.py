import polars as pl

from animeippo.providers import util


class StubMapper:
    def map(self, original):
        return pl.Series([f"{original} ran"])


def test_get_features():
    original = pl.DataFrame({"features1": ["1", "2", "3"], "features2": ["test", "test", "test"]})

    features = original.select(util.get_feature_selector(["features1", "features2"]))

    assert features[0].item().to_list() == ["1", "test"]


def test_mapping_skips_keys_not_in_dataframe():
    mapping = {"test1": StubMapper(), "test3": StubMapper()}

    actual = util.run_mappers("test1", mapping, {"test1": pl.Utf8})
    assert actual["test1"].to_list() == ["test1 ran"]
    assert "test3" not in actual.columns


def test_transformation_does_not_fail_with_empty_data():
    data = util.transform_to_animeippo_format(pl.DataFrame(), ["genres", "tags"], [], {})

    assert type(data) is pl.DataFrame
    assert len(data) == 0

    data = util.transform_to_animeippo_format(
        pl.DataFrame({"data": {"test": "test"}}), ["genres", "tags"], {}, {}
    )
    assert type(data) is pl.DataFrame
    assert len(data) == 0
