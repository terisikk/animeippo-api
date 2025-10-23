import abc

import polars as pl

from animeippo.analysis import statistics


class AbstractScorer(abc.ABC):
    def __init__(self, weight=1.0):
        self.weight = weight

    @abc.abstractmethod
    def score(self, scoring_target_df, compare_df):
        pass

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

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        # Get user feature weights (positive signal from correlations)
        user_positive_weights = statistics.weight_encoded_categoricals_correlation(
            compare_df, "encoded", data.all_features, header_name="features"
        )

        # Get negative signal from dropped/paused features
        user_negative_weights = statistics.weight_encoded_categoricals_correlation(
            compare_df.with_columns(
                dropped_or_paused=pl.col("user_status").is_in(["dropped", "paused"])
            ),
            "encoded",
            data.all_features,
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

        # === 2. COMPUTE DENOMINATORS ===
        feature_count_per_item = scoring_target_df["features"].list.len().fill_null(0)

        # diminishing returns for positive signal
        denominator_positive = (
            feature_count_per_item**self.GAMMA
            if self.GAMMA != 0.5  # noqa: PLR2004 Use sqrt method for square root
            else feature_count_per_item.sqrt()
        ).replace(0, 1.0)

        denominator_mean = feature_count_per_item.replace(0, 1.0)

        features_exploded = data.seasonal_explode_cached("features")

        # === 3. AGGREGATE SIGNALS ===
        positive_signal = (
            statistics.weighted_sum_for_categorical_values(
                features_exploded, "features", debiased_weights
            )
            / denominator_positive
        )

        negative_signal = (
            statistics.weighted_sum_for_categorical_values(
                features_exploded,
                "features",
                user_negative_weights.rename({"features": "name"}),
                fillna=0.0,
            )
            / denominator_mean
        )

        # catalogue baseline (expected mass from common features)
        catalogue_baseline = (
            statistics.weighted_sum_for_categorical_values(
                features_exploded,
                "features",
                catalogue_frequency.rename({"pc": "weight", "features": "name"}),
            )
            / denominator_mean
        )

        # === 4. COMBINE AND CLAMP ===
        raw_score = positive_signal - negative_signal - self.LAMBDA * catalogue_baseline
        clamped_score = raw_score.clip(lower_bound=0.0)

        return statistics.rank_series(clamped_score)


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

        # Penalize scores with no genres
        scores = scores - 0.05 * (scoring_target_df["genres"].list.len() == 0)

        return statistics.rank_series(scores)


class StudioCorrelationScorer(AbstractScorer):
    name = "studiocorrelationscore"

    def score(self, data):
        weights = data.user_profile.studio_correlations

        median = weights["weight"].median()

        scores = statistics.weighted_mean_for_categorical_values(
            data.seasonal_explode_cached("studios"), "studios", weights, median
        )

        return statistics.rank_series(scores)


class ClusterSimilarityScorer(AbstractScorer):
    name = "clusterscore"

    def __init__(self, weighted=False, weight=1.0):
        super().__init__(weight=weight)
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

        return statistics.rank_series(scores)


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

        return statistics.rank_series(scores["directscore"])


class PopularityScorer(AbstractScorer):
    name = "popularityscore"

    def score(self, data):
        scoring_target_df = data.seasonal

        scores = scoring_target_df["popularity"]

        return statistics.rank_series(scores.rank())


class ContinuationScorer(AbstractScorer):
    name = "continuationscore"

    BASE_CONTINUATION_BONUS = 1.5
    COMPLETION_BONUS = 1.0

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        user_baseline = statistics.mean_score_default(compare_df, 5.0)

        rdf = (
            scoring_target_df.explode("continuation_to")
            .select(["id", "continuation_to"])
            .join(
                compare_df.select(["id", "score", "user_status"]),
                left_on="continuation_to",
                right_on="id",
                how="left",
            )
            .with_columns(
                [
                    # Track whether this was a valid continuation match (exists in watchlist)
                    pl.col("user_status").is_not_null().alias("has_continuation"),
                    pl.when(pl.col("user_status") == "completed")
                    .then(pl.lit(1.0))
                    .when(pl.col("user_status").is_in(["watching", "paused"]))
                    .then(pl.lit(0.5))
                    .otherwise(pl.lit(0.2))
                    .alias("confidence"),
                ]
            )
            .with_columns(
                # Apply shrinkage: blend prior score towards baseline based on confidence
                # If score is null, use baseline
                shrunk_score=(
                    user_baseline
                    + pl.col("confidence")
                    * (pl.col("score").fill_null(user_baseline) - user_baseline)
                )
            )
            .with_columns(
                # Only apply bonuses for valid continuations
                continuationscore=pl.when(pl.col("has_continuation"))
                .then(
                    pl.col("shrunk_score")
                    + self.BASE_CONTINUATION_BONUS
                    + pl.when(pl.col("user_status") == "completed")
                    .then(pl.lit(self.COMPLETION_BONUS))
                    .otherwise(pl.lit(0.0))
                )
                .otherwise(pl.lit(None))
            )
            .group_by("id", maintain_order=True)
            .agg(pl.col("continuationscore").max())
        )

        return rdf["continuationscore"].fill_null(0.0).clip(0, 10) / 10


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

        BASE_PENALTY = {
            "TV": 0.0,
            "MOVIE": 0.0,
            "OVA": 0.05,
            "ONA": 0.05,
            "SPECIAL": 0.25,
            "TV_SHORT": 0.25,
            "MUSIC": 0.80,
            "ONE_SHOT": 0.50,
            "MANGA": 0.0,
            "NOVEL": 0.0,
        }

        CUTOFF = 0.75
        EP50 = scoring_target_df["episodes"].median() or 12
        DU50 = scoring_target_df["duration"].median() or 24
        SHORTNESS_PENALTY = 0.2

        out = (
            scoring_target_df.with_columns(
                penalty_base=pl.col("format")
                .replace_strict(BASE_PENALTY, default=0.05)
                .cast(pl.Float64),
            )
            .with_columns(
                penalty=(
                    pl.col("penalty_base")
                    + pl.when(
                        (
                            (pl.col("episodes") < CUTOFF * EP50) & (pl.col("format") != "MOVIE")
                        ).fill_null(False)
                    )
                    .then(SHORTNESS_PENALTY)
                    .otherwise(0.0)
                    + pl.when((pl.col("duration") < CUTOFF * DU50).fill_null(False))
                    .then(SHORTNESS_PENALTY)
                    .otherwise(0.0)
                ).cast(pl.Float64)
            )
            .select(pl.col("penalty").clip(lower_bound=0.0, upper_bound=1.0))
        )

        return out.get_column("penalty")


class DirectorCorrelationScorer(AbstractScorer):
    name = "directorcorrelationscore"

    def score(self, data):
        weights = data.user_profile.director_correlations

        median = weights["weight"].median()

        scores = statistics.weighted_mean_for_categorical_values(
            data.seasonal_explode_cached("directors"), "directors", weights, median
        )

        return statistics.rank_series(scores)
