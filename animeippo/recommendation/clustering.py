import sklearn.cluster as skcluster

import pandas as pd

from animeippo.recommendation import analysis


class AnimeClustering:
    def __init__(self):
        self.model = skcluster.AgglomerativeClustering(
            n_clusters=None, metric="precomputed", linkage="average", distance_threshold=0.85
        )

        self.n_clusters = None
        self.fit = False
        self.clustered_series = None

    def cluster_by_features(self, series, index):
        distances = pd.DataFrame(analysis.distance(series, series), index=index)

        clusters = self.model.fit_predict(distances)

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
            series,
            self.clustered_series["encoded"],
        )

        max_columns = similarities.idxmax(axis=1)

        nearest_clusters = max_columns.apply(lambda x: self.clustered_series.iloc[x]["cluster"])

        return nearest_clusters
