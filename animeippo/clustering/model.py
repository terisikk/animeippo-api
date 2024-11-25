import numpy as np
import sklearn.cluster as skcluster

from animeippo.analysis import statistics

from ..analysis import similarity


class AnimeClustering:
    """Wraps an sklearn or similar clustering model
    to some boilerplate to allow switching to one
    clustering model from one place.

    Also allows to do do crude cluster predictions
    for new data even when the underlying model does
    not directly support it.
    """

    def __init__(
        self,
        distance_metric="jaccard",
        distance_threshold=0.85,
        linkage="average",
        n_clusters=None,
        **kwargs,
    ):
        self.model = skcluster.AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=distance_metric,
            distance_threshold=distance_threshold,
            linkage=linkage,
            **kwargs,
        )

        self.n_clusters = None
        self.is_fit = False
        self.clustered_series = None
        self.distance_metric = distance_metric

    def cluster_by_features(self, dataframe):
        series = np.vstack(dataframe["encoded"].to_numpy())

        if self.distance_metric == "cosine":
            # Cosine is undefined for zero-vectors, need to hack (or change metric)
            clusters = np.full(len(series), -1)
            mask = series.sum(axis=1) > 0

            result = self.model.fit_predict(series[mask])

            if result is not None:
                clusters[mask] = result
            else:
                clusters = None
        else:
            clusters = self.model.fit_predict(series)

        if clusters is not None:
            self.is_fit = True
            self.n_clusters = self.model.n_clusters_
            self.clustered_series = dataframe.with_columns(cluster=clusters)

        return clusters

    def predict(self, series, similarities=None):
        if not self.is_fit:
            raise RuntimeError("Cluster is not fitted yet. Please call cluster_by_features first.")

        if similarities is None:
            similarities = similarity.categorical_similarity(
                self.clustered_series["encoded"],
                series,
                metric=self.distance_metric,
            ).with_columns(id=self.clustered_series["id"])

        idymax = statistics.idymax(similarities)

        return idymax.join(
            self.clustered_series.select("id", "cluster"),
            left_on="idymax",
            right_on="id",
        )["cluster"]
