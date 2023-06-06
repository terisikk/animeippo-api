import pandas as pd
import numpy as np
import pytest

from animeippo.recommendation import clustering


class FaultyClusterStub:
    def fit_predict(self, series):
        return None


def test_predict_cannot_be_called_when_clustring_fails():
    model = clustering.AnimeClustering()

    model.model = FaultyClusterStub()

    series = pd.Series([[0, 1, 2], [1, 2, 3]])
    clusters = model.cluster_by_features(np.vstack(series), series.index)

    assert clusters is None

    with pytest.raises(RuntimeError):
        model.predict(series)


def test_predict_returns_cluster_of_the_most_similar_element():
    model = clustering.AnimeClustering()

    series = pd.Series([[True, True, False, False], [False, False, True, True]])
    clusters = model.cluster_by_features(np.vstack(series), series.index)

    actual = model.predict(pd.Series([[False, False, True, True]]))

    assert actual[0] == clusters[1]
