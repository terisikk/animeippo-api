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

    - Debiases common features via catalogue frequency
    - Applies diminishing returns to long feature lists
    - Combines positive signal, negative signal (dropped/paused), and catalogue baseline
    - Returns rank-normalized scores in [0, 1]
    """

    name = "featurecorrelationscore"

    BETA = 0.7  # Debias strength for common features (higher = more debiasing)
    LAMBDA = 0.25  # Catalogue baseline strength (residual penalty for common features)

    # Diminishing returns exponent for long feature lists.
    # Set GAMMA to < 0.5 for stronger diminishing returns for long lists
    # or > 0.5 for weaker diminishing returns
    GAMMA = 0.5

    # Numerical stability constant for avoiding division by zero
    # and stabilizing effect on very rare features
    EPSILON = 1e-6

    MIN_HISTORY_THRESHOLD = 20

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        # Get user feature weights (positive signal from correlations)
        user_positive_weights = statistics.weight_encoded_categoricals_correlation(
            compare_df, "encoded", header_name="features"
        )

        # Get negative signal from dropped/paused features
        user_negative_weights = statistics.weight_encoded_categoricals_correlation(
            compare_df.with_columns(
                dropped_or_paused=pl.col("user_status").is_in(["DROPPED", "PAUSED"])
            ),
            "encoded",
            against="dropped_or_paused",
            header_name="features",
        )

        # Compute catalogue frequency, i.e. P(feature)
        catalogue_frequency = statistics.catalogue_frequency(
            scoring_target_df, "features", value_col="features"
        )

        # === 1. DEBIAS POSITIVE WEIGHTS  ===
        debiased_weights = (
            user_positive_weights.join(
                catalogue_frequency.with_columns(pl.col("features").cast(pl.Utf8)),
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

        # === 2. COMPUTE DENOMINATOR ===
        feature_count_per_item = scoring_target_df["features"].list.len().fill_null(0)
        denominator = feature_count_per_item.replace(0, 1.0)

        features_exploded = data.seasonal_explode_cached("features")

        # === 3. AGGREGATE SIGNALS ===
        positive_signal = (
            statistics.weighted_sum_for_categorical_values(
                features_exploded, "features", debiased_weights
            )
            / denominator
        )

        negative_signal = (
            statistics.weighted_sum_for_categorical_values(
                features_exploded,
                "features",
                user_negative_weights.rename({"features": "name"}),
                fillna=0.0,
            )
            / denominator
        )

        # catalogue baseline (expected mass from common features)
        catalogue_baseline = (
            statistics.weighted_sum_for_categorical_values(
                features_exploded,
                "features",
                catalogue_frequency.rename({"pc": "weight", "features": "name"}),
            )
            / denominator
        )

        # === 4. COMBINE AND CLAMP ===
        raw_score = positive_signal - negative_signal - self.LAMBDA * catalogue_baseline
        clamped_score = raw_score.clip(lower_bound=0.0)

        history_confidence = min(len(compare_df) / self.MIN_HISTORY_THRESHOLD, 1.0)
        confidence = self.feature_confidence(scoring_target_df) * history_confidence

        return ScorerResult(statistics.rank_series(clamped_score), confidence)


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
                # Predecessor rating normalized to 0-1
                predecessor_rating=pl.col("score").fill_null(0).cast(pl.Float64) / 10.0,
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
