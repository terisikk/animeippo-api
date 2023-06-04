import abc
import numpy as np
import pandas as pd
import sklearn.cluster as skcluster

from animeippo.recommendation import analysis, encoding


class AbstractScorer(abc.ABC):
    @abc.abstractmethod
    def score(self, scoring_target_df, compare_df):
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass


class FeaturesSimilarityScorer(AbstractScorer):
    name = "featuresimilarityscore"

    def __init__(self, weighted=False):
        self.weighted = weighted

    def score(self, scoring_target_df, compare_df):
        all_features = analysis.unique_features_from_categoricals(
            compare_df["features"], scoring_target_df["features"]
        )

        encoder = encoding.CategoricalEncoder(all_features)

        scores = analysis.similarity_of_anime_lists(
            scoring_target_df["features"], compare_df["features"], encoder
        )

        if self.weighted:
            averages = analysis.mean_score_per_categorical(
                compare_df.explode("features"), "features"
            )
            weights = scoring_target_df["features"].apply(
                analysis.weighted_mean_for_categorical_values, args=(averages.fillna(0.0),)
            )
            scores = scores * weights

        return analysis.normalize_column(scores)


class GenreAverageScorer(AbstractScorer):
    name = "genreaveragescore"

    def score(self, scoring_target_df, compare_df):
        weights = analysis.weight_categoricals(compare_df, "genres")

        scores = scoring_target_df["genres"].apply(
            analysis.weighted_mean_for_categorical_values, args=(weights.fillna(0.0),)
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
            analysis.weighted_mean_for_categorical_values,
            args=(weights.fillna(weights.mode()[0]),),
        )

        return analysis.normalize_column(scores)


class ClusterSimilarityScorer(AbstractScorer):
    name = "clusterscore"

    def __init__(self, weighted=False):
        self.model = skcluster.AgglomerativeClustering(
            n_clusters=None, metric="precomputed", linkage="average", distance_threshold=0.85
        )

        self.weighted = weighted

    def score(self, scoring_target_df, compare_df):
        all_features = analysis.unique_features_from_categoricals(
            compare_df["features"], scoring_target_df["features"]
        )

        encoder = encoding.CategoricalEncoder(all_features)

        compare_df["cluster"], nclusters = analysis.cluster_by_features(
            compare_df, "features", encoder, self.model
        )

        scores = pd.DataFrame(index=scoring_target_df.index, columns=range(0, nclusters))

        print("CLUSTERS: ", nclusters)

        st_encoded = encoder.encode(scoring_target_df["features"])

        for cluster_id, cluster in compare_df.groupby("cluster"):
            cl_encoded = encoder.encode(cluster["features"])

            similarities = pd.DataFrame(
                analysis.similarity(st_encoded, cl_encoded),
                index=scoring_target_df.index,
            ).mean(axis=1, skipna=True)

            if self.weighted:
                averages = cluster["score"].mean()
                similarities = similarities * averages

            scores[cluster_id] = similarities

        if self.weighted:
            weights = np.sqrt(compare_df["cluster"].value_counts())
            scores = scores * weights

        return analysis.normalize_column(scores.apply(np.max, axis=1))


class DirectSimilarityScorer(AbstractScorer):
    name = "directscore"

    def score(self, scoring_target_df, compare_df):
        all_features = analysis.unique_features_from_categoricals(
            compare_df["features"], scoring_target_df["features"]
        )

        encoder = encoding.CategoricalEncoder(all_features)

        similarities = analysis.categorical_similarity(
            scoring_target_df["features"], compare_df["features"], encoder
        )

        max_values = similarities.max(axis=1)
        max_columns = similarities[similarities.ge(max_values, axis=0)]

        scores = max_values * max_columns.notna().apply(
            lambda row: compare_df.iloc[max_columns.columns[row]]["score"].mean(), axis=1
        ).fillna(6.0)

        return analysis.normalize_column(scores)


class PopularityScorer(AbstractScorer):
    name = "popularityscore"

    def score(self, scoring_target_df, compare_df):
        scores = scoring_target_df["popularity"]

        return analysis.normalize_column(scores.rank())


class ContinuationScorer(AbstractScorer):
    name = "continuationscore"

    SCALE_MIDDLE = 6

    def score(self, scoring_target_df, compare_df):
        mean_score = compare_df["score"].mean()

        rdf = scoring_target_df.explode("related_anime")

        rdf["score"] = np.nan

        for i, row in rdf.iterrows():
            related_index = row["related_anime"]
            is_continuation = related_index in compare_df.index
            rdf.at[i, "score"] = (
                compare_df.at[related_index, "score"] if is_continuation else self.SCALE_MIDDLE
            )

        rdf["score"] = rdf["score"].fillna(mean_score)

        scores = rdf.groupby("id")["score"].max()

        return scores / 10


class SourceScorer(AbstractScorer):
    name = "sourcescore"

    def score(self, scoring_target_df, compare_df):
        averages = compare_df.groupby("source")["score"].mean() / 10

        scores = scoring_target_df["source"].apply(lambda x: averages.get(x, 0))

        return analysis.normalize_column(scores)
