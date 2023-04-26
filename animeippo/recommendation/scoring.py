import abc
import numpy as np
import pandas as pd
import sklearn.preprocessing as skpre
import sklearn.cluster as skcluster

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

    def __init__(self, genre_tags, weighted=False):
        self.encoder = CategoricalEncoder(genre_tags)
        self.weighted = weighted

    def score(self, scoring_target_df, compare_df):
        scores = analysis.similarity_of_anime_lists(scoring_target_df, compare_df, self.encoder)

        if self.weighted:
            averages = analysis.mean_score_per_categorical(compare_df, "genres")
            weights = scoring_target_df["genres"].apply(
                analysis.weighted_mean_for_categorical_values, args=(averages,)
            )
            scores = scores * weights

        return analysis.normalize_column(scores)


# Better than GenreSimilarityScorer
class GenreAverageScorer(AbstractScorer):
    name = "genreaveragescore"

    def score(self, scoring_target_df, compare_df):
        weights = analysis.weight_categoricals(compare_df, "genres")

        scores = scoring_target_df["genres"].apply(
            analysis.weighted_mean_for_categorical_values, args=(weights,)
        )

        return analysis.normalize_column(scores)


# This gives way too much zero. Replace with mean / mode or just use the better averagescorer.
class StudioCountScorer(AbstractScorer):
    name = "studiocountscore"

    def score(self, scoring_target_df, compare_df):
        counts = compare_df.explode("studios")["studios"].value_counts()

        scores = scoring_target_df.apply(self.max_studio_count, axis=1, args=(counts,))

        return analysis.normalize_column(scores)

    def max_studio_count(self, row, counts):
        if len(row["studios"]) == 0:
            return 0.0

        return np.max([counts.get(studio, 0.0) for studio in row["studios"]])


class StudioAverageScorer:
    name = "studioaveragescore"

    def score(self, scoring_target_df, compare_df):
        weights = analysis.weight_categoricals(compare_df, "studios")

        scores = scoring_target_df["studios"].apply(
            analysis.weighted_mean_for_categorical_values, args=(weights, weights.mode()[0])
        )

        return analysis.normalize_column(scores)


class ClusterSimilarityScorer(AbstractScorer):
    name = "clusterscore"

    def __init__(self, genre_tags, weighted=False):
        self.model = skcluster.AgglomerativeClustering(
            n_clusters=None, metric="precomputed", linkage="average", distance_threshold=0.8
        )

        self.weighted = weighted
        self.encoder = CategoricalEncoder(genre_tags)

    def score(self, scoring_target_df, compare_df):
        scores = pd.DataFrame(index=scoring_target_df.index)

        encoded = self.encoder.encode(compare_df["genres"])
        distances = pd.DataFrame(1 - analysis.similarity(encoded, encoded), index=compare_df.index)

        compare_df["cluster"] = self.model.fit_predict(distances)

        for cluster_id, cluster in compare_df.groupby("cluster"):
            similarities = analysis.similarity_of_anime_lists(
                scoring_target_df, cluster, self.encoder
            )

            if self.weighted:
                averages = cluster["score"].mean()
                similarities = similarities * averages

            scores[cluster_id] = similarities

        if self.weighted:
            weights = np.sqrt(compare_df["cluster"].value_counts())
            scores = scores * weights

        return analysis.normalize_column(scores.apply(np.max, axis=1))


class PopularityScorer(AbstractScorer):
    name = "popularityscore"

    def score(self, scoring_target_df, compare_df):
        scores = scoring_target_df["num_list_users"]

        return analysis.normalize_column(scores)


class CategoricalEncoder:
    def __init__(self, classes):
        self.classes = classes
        self.mlb = skpre.MultiLabelBinarizer(classes=classes)
        self.mlb.fit(None)

    def encode(self, series, dtype=bool):
        return np.array(self.mlb.transform(series), dtype=dtype)
