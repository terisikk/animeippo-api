import abc
from typing import NamedTuple

import polars as pl

from animeippo.analysis import statistics


class ScorerResult(NamedTuple):
    score: pl.Series
    confidence: pl.Series


class AbstractScorer(abc.ABC):
    MIN_FEATURE_THRESHOLD = 4

    def __init__(self, weight=1.0):
        self.weight = weight

    @abc.abstractmethod
    def score(self, scoring_target_df, compare_df):
        pass

    def feature_confidence(self, dataframe):
        """Confidence based on feature richness of candidates."""
        return (dataframe["features"].list.len().fill_null(0) / self.MIN_FEATURE_THRESHOLD).clip(
            upper_bound=1.0
        )

    @property
    @abc.abstractmethod
    def name(self):
        pass


class FeatureCorrelationScorer(AbstractScorer):
    """Score items based on user feature correlations with debiasing.

    Pipeline stages:
    1. Debias positive weights by catalogue frequency
    2. Compute denominator with diminishing returns for long feature lists
    3. Aggregate positive, negative, and catalogue baseline signals
    4. Combine and clamp to produce final score
    5. Compute confidence from feature richness, history size, and contested features
    """

    name = "featurecorrelationscore"

    BETA = 0.7  # Debias strength (higher = more debiasing of common features)
    LAMBDA = 0.25  # Catalogue baseline penalty strength
    GAMMA = 0.5  # Diminishing returns exponent for feature list length
    EPSILON = 1e-6  # Numerical stability constant
    MIN_HISTORY_THRESHOLD = 20
    CONTESTED_THRESHOLD = 0.1  # Min weight to consider a feature "active"
    CONTESTED_PENALTY = 0.5  # Max confidence reduction from contested features

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        positive_weights = self.get_positive_weights(compare_df)
        negative_weights = self.get_negative_weights(compare_df)
        catalogue_freq = self.get_catalogue_frequency(scoring_target_df)

        debiased = self.debias_weights(positive_weights, catalogue_freq)
        denominator = self.get_denominator(scoring_target_df)
        features_exploded = data.seasonal_explode_cached("features")

        positive_signal = self.aggregate_signal(features_exploded, debiased) / denominator
        negative_signal = (
            self.aggregate_signal(
                features_exploded,
                negative_weights.rename({"features": "name"}),
                fillna=0.0,
            )
            / denominator
        )
        baseline = (
            self.aggregate_signal(
                features_exploded,
                catalogue_freq.rename({"pc": "weight", "features": "name"}),
            )
            / denominator
        )

        raw_score = positive_signal - negative_signal - self.LAMBDA * baseline
        clamped_score = raw_score.clip(lower_bound=0.0)

        confidence = self.compute_confidence(
            scoring_target_df, compare_df, positive_weights, negative_weights
        )

        return ScorerResult(statistics.rank_series(clamped_score), confidence)

    def get_positive_weights(self, compare_df):
        return statistics.weight_encoded_categoricals_correlation(
            compare_df, "encoded", header_name="features"
        )

    def get_negative_weights(self, compare_df):
        return statistics.weight_encoded_categoricals_correlation(
            compare_df.with_columns(
                dropped_or_paused=pl.col("user_status").is_in(["DROPPED", "PAUSED"])
            ),
            "encoded",
            against="dropped_or_paused",
            header_name="features",
        )

    def get_catalogue_frequency(self, scoring_target_df):
        return statistics.catalogue_frequency(scoring_target_df, "features", value_col="features")

    def debias_weights(self, positive_weights, catalogue_freq):
        return (
            positive_weights.join(
                catalogue_freq.with_columns(pl.col("features").cast(pl.Utf8)),
                on="features",
                how="left",
            )
            .with_columns(
                pl.col("pc").fill_null(self.EPSILON),
                debiased_weight=(pl.col("weight") / (pl.col("pc") ** self.BETA + self.EPSILON)),
            )
            .select(["features", "debiased_weight"])
            .rename({"debiased_weight": "weight", "features": "name"})
        )

    def get_denominator(self, scoring_target_df):
        feature_count = scoring_target_df["features"].list.len().fill_null(0)
        return (feature_count**self.GAMMA).replace(0, 1.0)

    def aggregate_signal(self, features_exploded, weights, fillna=None):
        kwargs = {"fillna": fillna} if fillna is not None else {}
        return statistics.weighted_sum_for_categorical_values(
            features_exploded, "features", weights, **kwargs
        )

    def compute_confidence(self, scoring_target_df, compare_df, positive_weights, negative_weights):
        feature_conf = self.feature_confidence(scoring_target_df)
        history_conf = min(len(compare_df) / self.MIN_HISTORY_THRESHOLD, 1.0)

        contested_penalty = self.get_contested_penalty(
            scoring_target_df, positive_weights, negative_weights
        )

        return feature_conf * history_conf * contested_penalty

    def get_contested_penalty(self, scoring_target_df, positive_weights, negative_weights):
        """Reduce confidence when features have both strong positive and negative signals."""
        contested = positive_weights.join(negative_weights, on="features", suffix="_neg").filter(
            (pl.col("weight").abs() > self.CONTESTED_THRESHOLD)
            & (pl.col("weight_neg").abs() > self.CONTESTED_THRESHOLD)
        )

        contested_features = set(contested["features"].to_list())
        if not contested_features:
            return 1.0

        feature_counts = scoring_target_df["features"].list.len().fill_null(0)
        contested_counts = (
            scoring_target_df["features"]
            .list.eval(pl.element().is_in(list(contested_features)).sum())
            .list.first()
            .fill_null(0)
        )

        contested_ratio = contested_counts / feature_counts.replace(0, 1)

        return 1.0 - contested_ratio * self.CONTESTED_PENALTY


class StudioCorrelationScorer(AbstractScorer):
    name = "studiocorrelationscore"

    MIN_STUDIO_HISTORY = 3

    def score(self, data):
        weights = data.user_profile.studio_correlations

        median = weights["weight"].median()

        scores = statistics.weighted_mean_for_categorical_values(
            data.seasonal_explode_cached("studios"), "studios", weights, median
        )

        user_studios = set(weights["name"].to_list())
        studio_match_count = (
            data.seasonal["studios"]
            .list.eval(pl.element().is_in(list(user_studios)).sum())
            .list.first()
            .fill_null(0)
        )
        confidence = (studio_match_count / self.MIN_STUDIO_HISTORY).clip(upper_bound=1.0)

        return ScorerResult(statistics.rank_series(scores), confidence)


class ClusterSimilarityScorer(AbstractScorer):
    name = "clusterscore"

    def __init__(self, weight=1.0):
        super().__init__(weight=weight)

    def score(self, data):
        compare_df = data.watchlist

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

        confidence = self.feature_confidence(data.seasonal)

        return ScorerResult(statistics.rank_series(scores), confidence)


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
        idymax = idymax.with_columns(
            max=pl.col("max").fill_nan(None).fill_null(pl.col("max").mean())
        )

        scores = idymax.join(
            compare_df.select("id", "directscore"), left_on="idymax", right_on="id", how="left"
        )

        scores = scores.select(pl.col("directscore") * pl.col("max"))

        confidence = self.feature_confidence(data.seasonal)

        return ScorerResult(statistics.rank_series(scores["directscore"]), confidence)


class PopularityScorer(AbstractScorer):
    name = "popularityscore"

    MIN_MEMBER_COUNT = 1000
    MIN_SCORE = 50.0
    MAX_SCORE = 90.0
    HYPE_CONFIDENCE = 0.3

    def score(self, data):
        scoring_target_df = data.seasonal

        has_score = scoring_target_df["mean_score"].is_not_null()
        popularity = scoring_target_df["popularity"].fill_null(0).cast(pl.Float64)

        score_range = self.MAX_SCORE - self.MIN_SCORE
        rated_score = (
            (scoring_target_df["mean_score"].fill_null(0).cast(pl.Float64) - self.MIN_SCORE)
            / score_range
        ).clip(lower_bound=0.0, upper_bound=1.0)

        # Hype mode: normalized member count as signal for unrated shows
        hype_score = (popularity / self.MIN_MEMBER_COUNT).clip(upper_bound=1.0)

        rated_confidence = (popularity / self.MIN_MEMBER_COUNT).clip(upper_bound=1.0)

        result = scoring_target_df.select(
            score=pl.when(has_score).then(rated_score).otherwise(hype_score),
            confidence=pl.when(has_score)
            .then(rated_confidence)
            .otherwise(pl.lit(self.HYPE_CONFIDENCE)),
        )

        return ScorerResult(result["score"], result["confidence"])


class ContinuationScorer(AbstractScorer):
    name = "continuationscore"

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        user_mean = statistics.mean_score_default(compare_df, 5.0)

        joined = (
            scoring_target_df.explode("continuation_to")
            .select(["id", "continuation_to"])
            .join(
                compare_df.select(["id", "score", "user_status"]),
                left_on="continuation_to",
                right_on="id",
                how="left",
            )
            .with_columns(
                predecessor_rating=pl.col("score").fill_null(user_mean).cast(pl.Float64) / 10.0,
                # Completion status weight
                completion_weight=pl.col("user_status")
                .replace_strict(
                    {"COMPLETED": 1.0, "CURRENT": 0.7, "PAUSED": 0.3, "DROPPED": 0.1},
                    default=0.0,
                )
                .cast(pl.Float64),
            )
            .with_columns(
                # Strength = how much the user engaged with the predecessor
                strength=pl.col("predecessor_rating") * pl.col("completion_weight"),
            )
            .group_by("id", maintain_order=True)
            .agg(pl.col("strength").max())
        )

        score = joined["strength"].fill_null(0.0)
        confidence = score

        return ScorerResult(score, confidence)


class AdaptationScorer(AbstractScorer):
    name = "adaptationscore"

    DEFAULT_SCORE = 0
    DEFAULT_MEAN_SCORE = 5

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.mangalist
        n = len(scoring_target_df)

        if compare_df is None:
            return ScorerResult(
                pl.Series([0.0] * n),
                pl.Series([0.0] * n),
            )

        mean_score = statistics.mean_score_default(compare_df, self.DEFAULT_MEAN_SCORE)

        rdf = scoring_target_df.explode("adaptation_of")

        rdf = (
            rdf.select(["id", "adaptation_of"])
            .join(
                compare_df.with_columns(
                    pl.col("score").alias("adaptationscore").fill_null(mean_score),
                    # 1.0 if user rated, 0.5 if reading but unrated, 0.0 if no match
                    rating_confidence=pl.when(pl.col("score").is_not_null())
                    .then(pl.lit(1.0))
                    .otherwise(pl.lit(0.5)),
                ).select(["id", "adaptationscore", "rating_confidence"]),
                left_on="adaptation_of",
                right_on="id",
                how="left",
            )
            .with_columns(
                pl.col("adaptationscore").fill_null(self.DEFAULT_SCORE),
                pl.col("rating_confidence").fill_null(0.0),
            )
        )

        grouped = rdf.group_by("id", maintain_order=True).agg(
            pl.col("adaptationscore").max(),
            pl.col("rating_confidence").max(),
        )

        score = grouped["adaptationscore"] / 10
        confidence = grouped["rating_confidence"]

        return ScorerResult(score, confidence)
