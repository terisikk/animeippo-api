import abc
import numpy as np
import pandas as pd
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

            weights = scoring_target_df["features"].map_elements(
                lambda row: analysis.weighted_mean_for_categorical_values(
                    row, averages.fill_nan(0.0)
                )
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

        scores = scoring_target_df["features"].map_elements(
            # For once, np.sum is the wanted metric,
            # so that titles with only a few features get lower scores
            # and titles with multiple good features get on top.
            # Slightly diminished effect with sqrt.
            lambda row: analysis.weighted_sum_for_categorical_values(
                row, score_correlations
            )  # how about first map and then nanmean for whole series?
        ) / np.sqrt(scoring_target_df["features"].list.len())

        scores = scores - (
            scoring_target_df["features"].map_elements(
                lambda row: analysis.weighted_mean_for_categorical_values(row, drop_correlations)
            )
        )

        return analysis.normalize_column(scores)


class GenreAverageScorer(AbstractScorer):
    name = "genreaveragescore"

    def score(self, data):
        scoring_target_df = data.seasonal

        weights = analysis.weight_categoricals(data.watchlist_explode_cached("genres"), "genres")

        scores = scoring_target_df["genres"].apply(
            lambda row: analysis.weighted_sum_for_categorical_values(row, weights.fill_nan(0.0))
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

        scores = scoring_target_df["studios"].apply(
            lambda row: analysis.weighted_mean_for_categorical_values(row, weights.fill_null(mode))
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

        st_encoded = np.stack(scoring_target_df["encoded"])

        scores = pl.DataFrame()

        cluster_groups = compare_df.groupby("cluster")

        for cluster_id, cluster in cluster_groups:
            similarities = np.nanmean(
                analysis.similarity(
                    st_encoded,
                    np.stack(cluster["encoded"]),
                    metric=self.distance_metric,
                ),
                axis=1,
            )

            if self.weighted:
                averages = cluster["score"].mean() or 5
                similarities = similarities * averages

            scores = scores.with_columns(**{str(cluster_id): similarities})

        if self.weighted:
            scores = pl.DataFrame().with_columns(
                **{
                    str(key): scores[str(key)]
                    * compare_df["cluster"]
                    .value_counts()
                    .filter(pl.col("cluster") == key)["count"]
                    .sqrt()
                    .item()
                    for key in range(-1, 8)
                }
            )

        return analysis.normalize_column(scores.max_horizontal())


class DirectSimilarityScorer(AbstractScorer):
    name = "directscore"

    def __init__(self, distance_metric=None):
        self.distance_metric = distance_metric or "jaccard"

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        similarities = analysis.categorical_similarity(
            scoring_target_df["encoded"],
            compare_df.filter(~pl.col("id").is_in(scoring_target_df["id"]))["encoded"],
            metric=self.distance_metric,
        )

        with_idxmax = scoring_target_df.with_columns(
            idxmax=pl.Series(similarities.rows()).list.arg_max()
        )
        scores = with_idxmax.select([pl.col("idxmax").cast(pl.UInt32)]).join(
            compare_df.with_row_count().select(["row_nr", "score"]),
            left_on="idxmax",
            right_on="row_nr",
            how="left",
        )["score"]

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

        rdf = rdf.join(
            compare_df.with_columns(pl.col("score").fill_null(mean_score)).select("id", "score"),
            left_on="continuation_to",
            right_on="id",
            how="left",
        ).fill_null(self.DEFAULT_SCORE)

        return self.get_max_score_of_duplicate_relations(rdf, "score")["score"] / 10

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

        rdf = rdf.join(
            compare_df.with_columns(pl.col("score").fill_null(mean_score)).select("id", "score"),
            left_on="adaptation_of",
            right_on="id",
            how="left",
        ).fill_null(self.DEFAULT_SCORE)

        return self.get_max_score_of_duplicate_relations(rdf, "score")["score"] / 10

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

        weights = data.user_profile.director_correlations.cast(pl.Int64)

        mode = weights["weight"].mode()

        mode = mode[0] if len(mode) > 0 else mode

        scores = scoring_target_df["directors"].map_elements(
            lambda row: analysis.weighted_mean_for_categorical_values(row, weights.fill_null(mode)),
        )

        return analysis.normalize_column(scores)
