import abc
import numpy as np
import pandas as pd
import kmodes.kmodes as kmcluster
import kmodes.util.dissim as kdissim
import sklearn.preprocessing as skpre

import animeippo.recommendation.util as pdutil

from animeippo.recommendation import analysis


class AbstractScorer(abc.ABC):
    @abc.abstractmethod
    def score(self, scoring_target_df, compare_df):
        pass


class GenreSimilarityScorer(AbstractScorer):
    def __init__(self, encoder, weighted=False):
        self.encoder = encoder
        self.weighted = weighted

    def score(self, scoring_target_df, compare_df):
        scores = analysis.similarity_of_anime_lists(scoring_target_df, compare_df, self.encoder)

        if self.weighted:
            averages = analysis.mean_score_for_categorical_values(compare_df, "genres")
            weights = scoring_target_df["genres"].apply(
                analysis.weigh_by_user_score, args=(averages,)
            )
            scores = scores * weights

        return pdutil.normalize_column(scores)


class StudioSimilarityScorer(AbstractScorer):
    def __init__(self, weighted=False):
        self.weighted = weighted

    def score(self, scoring_target_df, compare_df):
        counts = compare_df.explode("studios")["studios"].value_counts()
        scores = []

        for i, row in scoring_target_df.iterrows():
            max = 0

            for studio in row["studios"]:
                count = counts.get(studio, 0.0)
                if count > max:
                    max = count

            scores.append(max)

        if self.weighted:
            averages = analysis.mean_score_for_categorical_values(compare_df, "studios")
            weigths = scoring_target_df["studios"].apply(
                analysis.weigh_by_user_score, args=(averages,)
            )

            scores = scores * weigths

        return pdutil.normalize_column(pd.Series(scores))


class ClusterSimilarityScorer(AbstractScorer):
    def __init__(self, encoder, clusters=10, weighted=False):
        self.model = kmcluster.KModes(n_clusters=clusters, cat_dissim=kdissim.ng_dissim, n_init=50)

        self.weighted = weighted
        self.encoder = encoder

    def score(self, scoring_target_df, compare_df):
        scores = pd.DataFrame(index=scoring_target_df.index)

        compare_df["cluster"] = self.model.fit_predict(self.encoder.encode(compare_df["genres"]))

        for cluster_id, cluster in compare_df.groupby("cluster"):
            similarities = analysis.similarity_of_anime_lists(
                scoring_target_df, cluster, self.encoder
            )

            if self.weighted:
                averages = cluster["user_score"].mean()
                similarities = similarities * averages

            scores["cluster_" + str(cluster_id)] = similarities

        return pdutil.normalize_column(scores.apply(np.max, axis=1))


class CategoricalEncoder:
    def __init__(self, classes):
        self.mlb = skpre.MultiLabelBinarizer(classes=classes)
        self.mlb.fit(None)

    def encode(self, series):
        return np.array(self.mlb.transform(series), dtype=bool)
