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
            averages = analysis.mean_score_for_categorical_values(compare_df, "genres")
            weights = scoring_target_df["genres"].apply(
                analysis.weigh_by_user_score, args=(averages,)
            )
            scores = scores * weights

        return pdutil.normalize_column(scores)


class GenreAverageScorer(AbstractScorer):
    name = "genreaveragescore"

    def __init__(self, encoder):
        self.encoder = encoder

    def score(self, scoring_target_df, compare_df):
        averages = analysis.mean_score_for_categorical_values(compare_df, "genres")
        averages = averages / 10

        weights = []

        counts = compare_df.explode("genres")["genres"].value_counts()
        counts = counts[counts.notnull()]

        for i, average in averages.items():
            weight = np.sqrt(counts[i])
            weights.append(weight * average)

        averages = pd.Series(weights, index=averages.index)

        scores = scoring_target_df.apply(score_from_genres, axis=1, args=(averages,))

        return pdutil.normalize_column(scores)


def score_from_genres(row, weighted_genre_scores):
    score = 0.0

    for genre in row["genres"]:
        score += weighted_genre_scores.get(genre, 0)

    return score / len(row["genres"])


# This gives way too much zero. Replace with mean / mode?
class StudioCountScorer(AbstractScorer):
    name = "studiocountscore"

    def score(self, scoring_target_df, compare_df):
        counts = compare_df.explode("studios")["studios"].value_counts()
        scores = pd.Series(index=scoring_target_df.index)

        for i, row in scoring_target_df.iterrows():
            max = 0

            for studio in row["studios"]:
                count = counts.get(studio, 0.0)
                if count > max:
                    max = count

            scores.at[i] = max

        return pdutil.normalize_column(scores)


# these seem to punish studios with unranked items, do like in animeplus and use mean for them
class StudioAverageScorer:
    name = "studioaveragescore"

    def __init__(self, weighted=False):
        self.weighted = weighted

    def score_old(self, scoring_target_df, compare_df):
        averages = analysis.mean_score_for_categorical_values(compare_df, "studios")
        scores = pd.Series(
            scoring_target_df["studios"].apply(analysis.weigh_by_user_score, args=(averages,)),
            index=scoring_target_df.index,
        )

        return pdutil.normalize_column(scores)

    def score(self, scoring_target_df, compare_df):
        averages = analysis.mean_score_for_categorical_values(compare_df, "studios")

        averages = averages[averages.notnull()]

        if self.weighted:
            counts = compare_df.explode("studios")["studios"].value_counts()
            counts = counts[counts.notnull()]

            weights = []

            for i, average in averages.items():
                weight = np.sqrt(counts[i])
                weights.append(weight * average)

            averages = pd.Series(weights, index=averages.index)

        scores = pd.Series(
            scoring_target_df["studios"].apply(analysis.weigh_by_user_score, args=(averages,)),
            index=scoring_target_df.index,
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
                averages = cluster["user_score"].mean()
                similarities = similarities * averages
                sizeweights.append(np.sqrt(len(cluster)))

            scores["cluster_" + str(cluster_id)] = similarities

        if self.weighted:
            sizeweights /= np.sum(sizeweights)
            scores = scores.mul(sizeweights, axis=1)

        return pdutil.normalize_column(scores.apply(np.max, axis=1))


class CategoricalEncoder:
    def __init__(self, classes):
        self.classes = classes
        self.mlb = skpre.MultiLabelBinarizer(classes=classes)
        self.mlb.fit(None)

    def encode(self, series, dtype=bool):
        return np.array(self.mlb.transform(series), dtype=dtype)
