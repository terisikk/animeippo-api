import sklearn.cluster as skcluster

import pandas as pd

from animeippo.recommendation import analysis


class AnimeClustering:
    def __init__(self, distance_metric="jaccard", distance_threshold=0.85, **kwargs):
        self.model = skcluster.AgglomerativeClustering(
            n_clusters=None, metric=distance_metric, distance_threshold=distance_threshold, **kwargs
        )

        self.n_clusters = None
        self.fit = False
        self.clustered_series = None
        self.distance_metric = distance_metric

    def cluster_by_features(self, series, index):
        clusters = self.model.fit_predict(series)

        if clusters is not None:
            self.fit = True
            self.n_clusters = self.model.n_clusters_
            self.clustered_series = pd.DataFrame(
                {"cluster": clusters, "encoded": series.tolist()}, index=index
            )

        return clusters

    def predict(self, series):
        if not self.fit:
            raise RuntimeError("Cluster is not fitted yet. Please call cluster_by_features first.")

        similarities = analysis.categorical_similarity(
            series, self.clustered_series["encoded"], metric=self.distance_metric
        )

        max_columns = similarities.idxmax(axis=1).fillna(-1).astype(int)

        nearest_clusters = max_columns.apply(lambda x: self.clustered_series.iloc[x]["cluster"])

        return nearest_clusters
