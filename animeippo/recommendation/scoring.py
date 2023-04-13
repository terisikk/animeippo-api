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

    @property
    @abc.abstractmethod
    def name(self):
        pass


class GenreSimilarityScorer(AbstractScorer):
    name = "genresimilarityscore"

    def __init__(self, encoder, weighted=False):
        self.encoder = encoder
        self.weighted = weighted

    def score(self, scoring_target_df, compare_df):
        scores = analysis.similarity_of_anime_lists(scoring_target_df, compare_df, self.encoder)

        if self.weighted:
            averages = analysis.mean_score_per_categorical(compare_df, "genres")
            weights = scoring_target_df["genres"].apply(
                analysis.weighted_mean_for_categorical_values, args=(averages,)
            )
            scores = scores * weights

        return pdutil.normalize_column(scores)


class GenreAverageScorer(AbstractScorer):
    name = "genreaveragescore"

    def score(self, scoring_target_df, compare_df):
        weights = analysis.weight_categoricals(compare_df, "genres")

        scores = scoring_target_df["genres"].apply(
            analysis.weighted_mean_for_categorical_values, args=(weights,)
        )

        return pdutil.normalize_column(pd.Series(scores))


# This gives way too much zero. Replace with mean / mode or just use the better averagescorer.
class StudioCountScorer(AbstractScorer):
    name = "studiocountscore"

    def score(self, scoring_target_df, compare_df):
        counts = compare_df.explode("studios")["studios"].value_counts()

        scores = scoring_target_df.apply(self.max_studio_count, axis=1, args=(counts,))

        return pdutil.normalize_column(scores)

    def max_studio_count(self, row, counts):
        return np.max([counts.get(studio, 0.0) for studio in row["studios"]])


class StudioAverageScorer:
    name = "studioaveragescore"

    def score(self, scoring_target_df, compare_df):
        weights = analysis.weight_categoricals(compare_df, "studios")

        scores = scoring_target_df["studios"].apply(
            analysis.weighted_mean_for_categorical_values, args=(weights, weights.mode()[0])
        )

        return pdutil.normalize_column(scores)


class ClusterSimilarityScorer(AbstractScorer):
    name = "clusterscore"

    def __init__(self, encoder, clusters=10, weighted=False):
        self.model = kmcluster.KModes(n_clusters=clusters, cat_dissim=kdissim.ng_dissim, n_init=50)

        self.weighted = weighted
        self.encoder = encoder

    def score(self, scoring_target_df, compare_df):
        scores = pd.DataFrame(index=scoring_target_df.index)

        compare_df["cluster"] = self.model.fit_predict(self.encoder.encode(compare_df["genres"]))

        sizeweights = []

        for cluster_id, cluster in compare_df.groupby("cluster"):
            similarities = analysis.similarity_of_anime_lists(
                scoring_target_df, cluster, self.encoder
            )

            if self.weighted:
                averages = cluster["score"].mean()
                similarities = similarities * averages
                sizeweights.append(np.sqrt(len(cluster)))

            scores["cluster_" + str(cluster_id)] = similarities

        if self.weighted:
            sizeweights /= np.sum(sizeweights)
            scores = scores.mul(sizeweights, axis=1)

        return pdutil.normalize_column(scores.apply(np.max, axis=1))


class PopularityScorer(AbstractScorer):
    name = "popularityscorer"

    def score(self, scoring_target_df, compare_df):
        scores = scoring_target_df["num_list_users"]

        return pdutil.normalize_column(scores)


class CategoricalEncoder:
    def __init__(self, classes):
        self.classes = classes
        self.mlb = skpre.MultiLabelBinarizer(classes=classes)
        self.mlb.fit(None)

    def encode(self, series, dtype=bool):
        return np.array(self.mlb.transform(series), dtype=dtype)
