import abc
import numpy as np
import pandas as pd
import kmodes.kmodes as kmcluster
import kmodes.util.dissim as kdissim

import animeippo.recommendation.util as pdutil

from animeippo.recommendation import analysis


class AbstractScorer(abc.ABC):
    @abc.abstractmethod
    def score(self, scoring_target_df, compare_df, encoder):
        pass


class GenreSimilarityScorer(AbstractScorer):
    def __init__(self, weighted=False):
        self.weighted = weighted

    def score(self, scoring_target_df, compare_df, encoder):
        similarities = analysis.similarity_of_anime_lists(scoring_target_df, compare_df, encoder)

        if self.weighted:
            averages = analysis.genre_average_scores(compare_df)
            similarities = pdutil.normalize_column(similarities) + (
                1.5
                * pdutil.normalize_column(
                    scoring_target_df["genres"].apply(analysis.user_genre_weight, args=(averages,))
                )
            )
            similarities = similarities / 2

        scoring_target_df["recommend_score"] = similarities

        return scoring_target_df.sort_values("recommend_score", ascending=False)


class ClusterSimilarityScorer(AbstractScorer):
    def __init__(self, n_clusters=10, weighted=False):
        self.model = kmcluster.KModes(
            n_clusters=n_clusters, cat_dissim=kdissim.ng_dissim, n_init=50
        )

        self.weighted = weighted

    def score(self, scoring_target_df, compare_df, encoder):
        scores = pd.DataFrame(index=scoring_target_df.index)

        compare_df["cluster"] = self.model.fit_predict(encoder.encode(compare_df["genres"]))

        for cluster_id, cluster in compare_df.groupby("cluster"):
            similarities = analysis.similarity_of_anime_lists(scoring_target_df, cluster, encoder)

            if self.weighted:
                averages = cluster["user_score"].mean() / 10
                similarities = similarities * averages

            scores["cluster_" + str(cluster_id)] = similarities

        scoring_target_df["recommend_score"] = scores.apply(np.max, axis=1)

        return scoring_target_df.sort_values("recommend_score", ascending=False)
