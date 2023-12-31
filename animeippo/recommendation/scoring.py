import abc
import math

import numpy as np
import polars as pl

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
            scoring_target_df["encoded"],
            compare_df.filter(~pl.col("id").is_in(scoring_target_df["id"]))["encoded"],
            self.distance_metric,
        )

        if self.weighted:
            averages = analysis.mean_score_per_categorical(
                data.watchlist_explode_cached("features"), "features"
            )
            averages.columns = ["features", "weight"]

            weights = analysis.weighted_mean_for_categorical_values(
                compare_df, "features", averages
            )

            scores = scores * weights

        return analysis.normalize_column(scores)


class FeatureCorrelationScorer(AbstractScorer):
    name = "featurecorrelationscore"

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        score_correlations = analysis.weight_encoded_categoricals_correlation(
            compare_df, "encoded", data.all_features
        )

        dropped_or_paused = compare_df["user_status"].is_in(["dropped", "paused"])

        drop_correlations = analysis.weight_encoded_categoricals_correlation(
            compare_df, "encoded", data.all_features, dropped_or_paused
        )

        scores = analysis.weighted_sum_for_categorical_values(
            scoring_target_df, "features", score_correlations
        ) / np.sqrt(scoring_target_df["features"].list.len())

        scores = scores - (
            analysis.weighted_mean_for_categorical_values(
                scoring_target_df, "features", drop_correlations
            )
        )

        return analysis.normalize_column(scores)


class GenreAverageScorer(AbstractScorer):
    name = "genreaveragescore"

    def score(self, data):
        scoring_target_df = data.seasonal

        weights = analysis.weight_categoricals(data.watchlist_explode_cached("genres"), "genres")

        scores = analysis.weighted_sum_for_categorical_values(
            scoring_target_df,
            "genres",
            weights,
        ) / np.sqrt(scoring_target_df["genres"].list.len())

        return analysis.normalize_column(scores)


# This gives way too much zero. Replace with mean / mode or just use the better studiocorrelationscore.
class StudioCountScorer(AbstractScorer):
    name = "studiocountscore"

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        counts = compare_df.explode("studios")["studios"].value_counts()
        counts = dict(zip(counts["studios"], counts["count"]))

        scores = scoring_target_df["studios"].map_elements(
            lambda row: self.max_studio_count(row, counts)
        )

        return analysis.normalize_column(scores)

    def max_studio_count(self, studios, counts):
        if len(studios) == 0:
            return 0.0

        return np.max([counts.get(studio, 0.0) for studio in studios])


class StudioCorrelationScorer:
    name = "studiocorrelationscore"

    def score(self, data):
        scoring_target_df = data.seasonal

        weights = data.user_profile.studio_correlations

        mode = weights["weight"].mode()

        mode = mode[0] if len(mode) > 0 else mode

        scores = analysis.weighted_mean_for_categorical_values(
            scoring_target_df, "studios", weights, mode
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

        scores = pl.DataFrame()

        similarities = data.similarity_matrix.join(
            compare_df.select("id", "cluster", "score"), how="left", on="id"
        )

        scores = (
            similarities.group_by("cluster", maintain_order=True)
            .agg(
                pl.exclude("cluster", "id", "score").mean()
                * pl.col("score").mean().fill_null(5)
                * pl.col("id").len().sqrt()
            )
            .select(pl.exclude("cluster"))
            .max()
            .transpose()
            .to_series()
        )

        return analysis.normalize_column(scores)


class DirectSimilarityScorer(AbstractScorer):
    name = "directscore"

    def __init__(self, distance_metric=None):
        self.distance_metric = distance_metric or "jaccard"

    def score(self, data):
        compare_df = data.watchlist

        compare_df = compare_df.with_columns(
            directscore=pl.col("score").fill_null(pl.col("score").mean())
        )

        similarities = data.similarity_matrix

        idymax = analysis.idymax(similarities)

        scores = idymax.join(
            compare_df.select("id", "directscore"), left_on="idymax", right_on="id"
        )["directscore"]

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
    DEFAULT_MEAN_SCORE = 5

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        mean_score = analysis.get_mean_score(compare_df, self.DEFAULT_MEAN_SCORE)

        rdf = scoring_target_df.explode("continuation_to")

        rdf = (
            rdf.select(["id", "continuation_to"])
            .join(
                compare_df.with_columns(
                    pl.col("score").alias("continuationscore").fill_null(mean_score)
                ).select(["id", "continuationscore"]),
                left_on="continuation_to",
                right_on="id",
                how="left",
            )
            .fill_null(self.DEFAULT_SCORE)
        )

        return (
            self.get_max_score_of_duplicate_relations(rdf, "continuationscore")["continuationscore"]
            / 10
        )

    def get_max_score_of_duplicate_relations(self, df, column):
        return df.group_by("id", maintain_order=True).agg(pl.col(column).max())


class AdaptationScorer(AbstractScorer):
    name = "adaptationscore"

    DEFAULT_SCORE = 0
    DEFAULT_MEAN_SCORE = 5

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.mangalist

        if compare_df is None:
            return None

        mean_score = analysis.get_mean_score(compare_df, self.DEFAULT_MEAN_SCORE)

        rdf = scoring_target_df.explode("adaptation_of")

        rdf = (
            rdf.select(["id", "adaptation_of"])
            .join(
                compare_df.with_columns(
                    pl.col("score").alias("adaptationscore").fill_null(mean_score)
                ).select(["id", "adaptationscore"]),
                left_on="adaptation_of",
                right_on="id",
                how="left",
            )
            .fill_null(self.DEFAULT_SCORE)
        )

        return (
            self.get_max_score_of_duplicate_relations(rdf, "adaptationscore")["adaptationscore"]
            / 10
        )

    def get_max_score_of_duplicate_relations(self, df, column):
        return df.group_by("id", maintain_order=True).agg(pl.col(column).max())


class SourceScorer(AbstractScorer):
    name = "sourcescore"

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        averages = compare_df.group_by("source", maintain_order=True).agg(pl.col("score").mean())

        scores = scoring_target_df.join(averages, on="source", how="left")["score"].fill_null(0)

        return analysis.normalize_column(scores)


class FormatScorer(AbstractScorer):
    name = "formatscore"

    FORMAT_MAPPING = {
        "TV": 1,
        "TV_SHORT": 0.8,
        "MOVIE": 1,
        "SPECIAL": 0.8,
        "OVA": 1,
        "ONA": 1,
        "MUSIC": 0.2,
        "MANGA": 1,
        "NOVEL": 1,
        "ONE_SHOT": 0.2,
    }

    def score(self, data):
        scoring_target_df = data.seasonal

        scores = scoring_target_df.with_columns(
            formatscore=pl.col("format").replace(self.FORMAT_MAPPING, default=1)
        )

        scores = scores.with_columns(
            formatscore=pl.when(
                (pl.col("episodes") < (0.75 * scoring_target_df["episodes"].median()))
            )
            .then(pl.col("formatscore") * 0.5)
            .when(pl.col("duration") < (0.76 * scoring_target_df["duration"].median()))
            .then(pl.col("formatscore") * 0.5)
            .otherwise(pl.col("formatscore"))
        )["formatscore"]

        return analysis.normalize_column(scores)

    def get_format_score(self, row, median_episodes, median_duration):
        score = self.FORMAT_MAPPING.get(row["format"], 1)

        if row.get("episodes", median_episodes) < (0.75 * median_episodes) and row.get(
            "duration", median_duration
        ) < (0.75 * median_duration):
            score = score * 0.5

        return score


class DirectorCorrelationScorer:
    name = "directorcorrelationscore"

    def score(self, data):
        scoring_target_df = data.seasonal

        weights = data.user_profile.director_correlations

        mode = weights["weight"].mode()

        mode = mode[0] if len(mode) > 0 else mode

        scores = analysis.weighted_mean_for_categorical_values(
            scoring_target_df, "directors", weights, mode
        )

        return analysis.normalize_column(scores)
