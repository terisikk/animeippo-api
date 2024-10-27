import abc

import polars as pl

from animeippo.analysis import statistics


class AbstractScorer(abc.ABC):
    @abc.abstractmethod
    def score(self, scoring_target_df, compare_df):
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass


class FeatureCorrelationScorer(AbstractScorer):
    name = "featurecorrelationscore"

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        score_correlations = statistics.weight_encoded_categoricals_correlation(
            compare_df, "encoded", data.all_features
        )

        drop_correlations = statistics.weight_encoded_categoricals_correlation(
            compare_df.with_columns(
                dropped_or_paused=pl.col("user_status").is_in(["dropped", "paused"])
            ),
            "encoded",
            data.all_features,
            "dropped_or_paused",
        )

        fdf = data.seasonal_explode_cached("features")

        scores = (
            statistics.weighted_sum_for_categorical_values(fdf, "features", score_correlations)
            / scoring_target_df["features"].list.len().sqrt()
        )

        scores = scores - (
            statistics.weighted_mean_for_categorical_values(fdf, "features", drop_correlations)
        )

        return statistics.normalize_series(scores)


class GenreAverageScorer(AbstractScorer):
    name = "genreaveragescore"

    def score(self, data):
        scoring_target_df = data.seasonal
        gdf = data.seasonal_explode_cached("genres")

        weights = statistics.weight_categoricals(data.watchlist_explode_cached("genres"), "genres")

        scores = (
            statistics.weighted_sum_for_categorical_values(
                gdf,
                "genres",
                weights,
            )
            / scoring_target_df["genres"].list.len().sqrt()
        )

        return statistics.normalize_series(scores)


class StudioCorrelationScorer:
    name = "studiocorrelationscore"

    def score(self, data):
        weights = data.user_profile.studio_correlations

        median = weights["weight"].median()

        scores = statistics.weighted_mean_for_categorical_values(
            data.seasonal_explode_cached("studios"), "studios", weights, median
        )

        return statistics.normalize_series(scores)


class ClusterSimilarityScorer(AbstractScorer):
    name = "clusterscore"

    def __init__(self, weighted=False):
        self.weighted = weighted

    def score(self, data):
        compare_df = data.watchlist

        scores = pl.DataFrame()

        similarities = data.get_similarity_matrix(filtered=True).join(
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

        return statistics.normalize_series(scores)


class DirectSimilarityScorer(AbstractScorer):
    name = "directscore"

    def score(self, data):
        compare_df = data.watchlist

        compare_df = compare_df.with_columns(
            directscore=pl.col("score").fill_null(pl.col("score").mean())
        )

        # Want to sort this so that argmax gives at least consistent results,
        # returning the index of the max score on invalid cases
        similarities = (
            data.get_similarity_matrix(filtered=True)
            .join(compare_df.select("id", "directscore"), on="id", how="left")
            .sort(["directscore", "id"], descending=[True, True])
            .drop("directscore")
        )

        idymax = statistics.idymax(similarities)

        # Feels kinda hackish, I think the max column should not have nans in the first place,
        # need to investigate why it does.
        idymax = idymax.with_columns(max=pl.col("max").fill_nan(None)).fill_null(
            pl.col("max").mean()
        )

        scores = idymax.join(
            compare_df.select("id", "directscore"), left_on="idymax", right_on="id", how="left"
        )

        return statistics.normalize_series(
            scores.select(pl.col("directscore") * pl.col("max"))["directscore"]
        )


class PopularityScorer(AbstractScorer):
    name = "popularityscore"

    def score(self, data):
        scoring_target_df = data.seasonal

        scores = scoring_target_df["popularity"]

        return statistics.normalize_series(scores.rank())


class ContinuationScorer(AbstractScorer):
    name = "continuationscore"

    DEFAULT_SCORE = 0
    DEFAULT_MEAN_SCORE = 5

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        mean_score = statistics.mean_score_default(compare_df, self.DEFAULT_MEAN_SCORE)

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

        mean_score = statistics.mean_score_default(compare_df, self.DEFAULT_MEAN_SCORE)

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


class FormatScorer(AbstractScorer):
    name = "formatscore"

    def score(self, data):
        scoring_target_df = data.seasonal

        FORMAT_MAPPING = {
            "TV": 1.0,
            "TV_SHORT": 0.8,
            "MOVIE": 1.0,
            "SPECIAL": 0.8,
            "OVA": 1.0,
            "ONA": 1.0,
            "MUSIC": 0.2,
            "MANGA": 1.0,
            "NOVEL": 1.0,
            "ONE_SHOT": 0.2,
        }

        CUTOFF = 0.75

        scores = scoring_target_df.with_columns(
            formatscore=pl.col("format").replace_strict(FORMAT_MAPPING, default=1)
        )

        episodes_median = scoring_target_df["episodes"].median()
        episodes_median = episodes_median if episodes_median is not None else 12

        duration_median = scoring_target_df["duration"].median()
        duration_median = duration_median if duration_median is not None else 24

        scores = scores.with_columns(
            formatscore=pl.when((pl.col("episodes") < (CUTOFF * episodes_median)))
            .then(pl.col("formatscore") * 0.5)
            .when(pl.col("duration") < (CUTOFF * duration_median))
            .then(pl.col("formatscore") * 0.5)
            .otherwise(pl.col("formatscore"))
        )["formatscore"]

        return statistics.normalize_series(scores)


class DirectorCorrelationScorer:
    name = "directorcorrelationscore"

    def score(self, data):
        weights = data.user_profile.director_correlations

        median = weights["weight"].median()

        scores = statistics.weighted_mean_for_categorical_values(
            data.seasonal_explode_cached("directors"), "directors", weights, median
        )

        return statistics.normalize_series(scores)
