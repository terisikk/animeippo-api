import pandas as pd
import numpy as np
import pytest

from animeippo.recommendation import clustering


class FaultyClusterStub:
    def fit_predict(self, series):
        return None


def test_clustering():
    model = clustering.AnimeClustering()

    series = pl.Series([[0, 1, 2], [1, 2, 3]])
    clusters = model.cluster_by_features(np.vstack(series), series.index)

    clusters = model.cluster_by_features(np.vstack(series), series.index)

    assert clusters.tolist() == [1, 0]


# Cosine is undefined for zero-vectors, so has a slightly different implementation
def test_clustering_with_cosine():
    model = clustering.AnimeClustering(distance_metric="cosine")

    series = pl.Series([[0, 0, 0], [1, 2, 3], [1, 2, 3]])
    clusters = model.cluster_by_features(np.vstack(series), series.index)

    clusters = model.cluster_by_features(np.vstack(series), series.index)

    assert clusters.tolist() == [-1, 0, 0]


def test_predict_cannot_be_called_when_clustering_fails():
    model = clustering.AnimeClustering(distance_metric="cosine")

    model.model = FaultyClusterStub()

    series = pl.Series([[0, 1, 2], [1, 2, 3]])
    clusters = model.cluster_by_features(np.vstack(series), series.index)

    assert clusters is None

    with pytest.raises(RuntimeError):
        model.predict(series)


def test_predict_returns_cluster_of_the_most_similar_element():
    model = clustering.AnimeClustering()

    series = pl.Series([[True, True, False, False], [False, False, True, True]])
    clusters = model.cluster_by_features(np.vstack(series), series.index)

    actual = model.predict(pl.Series([[False, False, True, True]]))

    assert actual[0] == clusters[1]
