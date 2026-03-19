import polars as pl
import pytest

from animeippo.clustering import model


class FaultyClusterStub:
    def fit_predict(self, series):
        return None


def test_clustering():
    ml = model.AnimeClustering(distance_threshold=0.33)

    # Use struct format (matching encoder output)
    series = pl.DataFrame({"encoded": [{"a": 0, "b": 1, "c": 2}, {"a": 1, "b": 2, "c": 3}]})
    clusters = ml.cluster_by_features(series)

    assert clusters.tolist() == [1, 0]


# Cosine is undefined for zero-vectors, so has a slightly different implementation
def test_clustering_with_cosine():
    ml = model.AnimeClustering(distance_metric="cosine")

    series = pl.DataFrame(
        {"encoded": [{"a": 0, "b": 0, "c": 0}, {"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2, "c": 3}]}
    )
    clusters = ml.cluster_by_features(series)

    assert clusters.tolist() == [-1, 0, 0]


def test_predict_cannot_be_called_when_clustering_fails():
    ml = model.AnimeClustering(distance_metric="cosine")

    ml.model = FaultyClusterStub()

    series = pl.DataFrame({"encoded": [{"a": 0, "b": 1, "c": 2}, {"a": 1, "b": 2, "c": 3}]})
    clusters = ml.cluster_by_features(series)

    assert clusters is None

    with pytest.raises(RuntimeError):
        ml.predict(pl.Series([{"a": 0, "b": 1, "c": 2}]))


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
