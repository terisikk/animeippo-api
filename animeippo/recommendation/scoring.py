import abc
import numpy as np
import pandas as pd

from animeippo.recommendation import analysis


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

    def __init__(self, weighted=False, distance_metric=None):
        self.weighted = weighted
        self.distance_metric = distance_metric or "jaccard"

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        scores = analysis.similarity_of_anime_lists(
            scoring_target_df["encoded"], compare_df["encoded"], self.distance_metric
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


class FeatureCorrelationScorer(AbstractScorer):
    name = "featurecorrelationscore"

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        score_correlations = analysis.weight_categoricals_correlation(
            compare_df, "encoded", data.all_features
        )

        scores = scoring_target_df["features"].apply(
            # For once, np.sum is the wanted metric, so that titles with only a few features get lower scores
            # and titles with multiple good features get on top. Slightly diminished effect with sqrt.
            analysis.weighted_sum_for_categorical_values,
            args=(score_correlations,),
        ) / np.sqrt(scoring_target_df["features"].str.len())

        return analysis.normalize_column(scores)


class GenreAverageScorer(AbstractScorer):
    name = "genreaveragescore"

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        weights = analysis.weight_categoricals(compare_df, "genres")

        scores = scoring_target_df["genres"].apply(
            analysis.weighted_sum_for_categorical_values, args=(weights.fillna(0.0),)
        ) / np.sqrt(scoring_target_df["genres"].str.len())

        return analysis.normalize_column(scores)


# This gives way too much zero. Replace with mean / mode or just use the better averagescorer.
class StudioCountScorer(AbstractScorer):
    name = "studiocountscore"

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        counts = compare_df.explode("studios")["studios"].value_counts()

        scores = scoring_target_df.apply(self.max_studio_count, axis=1, args=(counts,))

        return analysis.normalize_column(scores)

    def max_studio_count(self, row, counts):
        if len(row["studios"]) == 0:
            return 0.0

        return np.max([counts.get(studio, 0.0) for studio in row["studios"]])


class StudioAverageScorer:
    name = "studioaveragescore"

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        weights = analysis.weight_categoricals(compare_df, "studios")

        mode = weights.mode()

        mode = mode[0] if len(mode) > 0 else mode

        scores = scoring_target_df["studios"].apply(
            analysis.weighted_mean_for_categorical_values,
            args=(weights.fillna(mode),),
        )

        return analysis.normalize_column(scores)


class ClusterSimilarityScorer(AbstractScorer):
    name = "clusterscore"

    def __init__(self, weighted=False, distance_metric=None):
        self.weighted = weighted
        self.distance_metric = distance_metric or "jaccard"

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        st_encoded = np.vstack(scoring_target_df["encoded"])
        co_encoded = np.vstack(compare_df["encoded"])

        scores = pd.DataFrame(
            index=scoring_target_df.index, columns=range(0, len(compare_df["cluster"].unique()))
        )

        cluster_groups = compare_df.groupby("cluster")

        for cluster_id, cluster in cluster_groups:
            similarities = pd.DataFrame(
                analysis.similarity(
                    st_encoded,
                    co_encoded[cluster_groups.indices[cluster_id]],
                    metric=self.distance_metric,
                ),
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

    def __init__(self, distance_metric=None):
        self.distance_metric = distance_metric or "jaccard"

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        similarities = analysis.categorical_similarity(
            scoring_target_df["encoded"], compare_df["encoded"], metric=self.distance_metric
        )

        max_values = similarities.max(axis=1)
        max_columns = similarities[similarities.ge(max_values, axis=0)]

        scores = max_values * max_columns.notna().apply(
            lambda row: compare_df.loc[max_columns.columns[row]]["score"].mean(), axis=1
        ).fillna(6.0)

        return analysis.normalize_column(scores)


class PopularityScorer(AbstractScorer):
    name = "popularityscore"

    def score(self, data):
        scoring_target_df = data.seasonal

        scores = scoring_target_df["popularity"]

        return analysis.normalize_column(scores.rank())


class ContinuationScorer(AbstractScorer):
    name = "continuationscore"

    DEFAULT_SCORE = 0

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        mean_score = compare_df["score"].mean()

        rdf = scoring_target_df.explode("relations")

        rdf["score"] = np.nan

        for i, row in rdf.iterrows():
            related_index = row["relations"]
            is_continuation = related_index in compare_df.index
            rdf.at[i, "score"] = (
                compare_df.at[related_index, "score"] if is_continuation else self.DEFAULT_SCORE
            )

        rdf["score"] = rdf["score"].fillna(mean_score)

        scores = self.get_max_score_of_duplicate_relations(rdf)

        return scores / 10

    def get_max_score_of_duplicate_relations(self, rdf):
        return rdf.groupby(rdf.index)["score"].max()


class SourceScorer(AbstractScorer):
    name = "sourcescore"

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        averages = compare_df.groupby("source")["score"].mean() / 10

        scores = scoring_target_df["source"].apply(lambda x: averages.get(x, 0))

        return analysis.normalize_column(scores)
