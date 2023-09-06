import sklearn.cluster as skcluster

import pandas as pd
import numpy as np

from animeippo.recommendation import analysis


class AnimeClustering:
    def __init__(
        self,
        distance_metric="jaccard",
        distance_threshold=0.85,
        linkage="average",
        n_clusters=None,
        **kwargs
    ):
        self.model = skcluster.AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=distance_metric,
            distance_threshold=distance_threshold,
            linkage=linkage,
            **kwargs
        )

        self.n_clusters = None
        self.fit = False
        self.clustered_series = None
        self.distance_metric = distance_metric

    def cluster_by_features(self, series, index):
        # Cosine is undefined for zero-vectors, need to hack (or change metric)
        clusters = self.model.fit_predict(self.remove_rows_with_no_features(series))
        clusters = self.reinsert_rows_with_no_features_as_a_new_cluster(clusters, series)

        if clusters is not None:
            self.fit = True
            self.n_clusters = self.model.n_clusters_
            self.clustered_series = pd.DataFrame(
                {"cluster": clusters, "encoded": series.tolist()}, index=index
            )

        return clusters

    def remove_rows_with_no_features(self, series):
        return series[series.sum(axis=1) > 0]

    def reinsert_rows_with_no_features_as_a_new_cluster(self, clusters, series):
        if clusters is None:
            return None

        return np.insert(clusters, np.where(series.sum(axis=1) == 0)[0], -1)

    def predict(self, series):
        if not self.fit:
            raise RuntimeError("Cluster is not fitted yet. Please call cluster_by_features first.")

        similarities = analysis.categorical_similarity(
            series, self.clustered_series["encoded"], metric=self.distance_metric
        )

        max_columns = similarities.idxmax(axis=1).fillna(-1).astype(int)

        nearest_clusters = max_columns.apply(lambda x: self.clustered_series.iloc[x]["cluster"])

        return nearest_clusters
