import abc
from typing import ClassVar, NamedTuple

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
            .list.filter(pl.element().is_in(contested_features))
            .list.len()
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

        studio_match_count = (
            data.seasonal["studios"]
            .list.filter(pl.element().is_in(weights["name"].to_list()))
            .list.len()
            .fill_null(0)
        )
        confidence = (studio_match_count / self.MIN_STUDIO_HISTORY).clip(upper_bound=1.0)

        return ScorerResult(statistics.rank_series(scores), confidence)


class ClusterSimilarityScorer(AbstractScorer):
    name = "clusterscore"

    TOP_N = 3
    DECAY_WEIGHTS: ClassVar[list[float]] = [1.0, 0.5, 0.25]

    def score(self, data):
        compare_df = data.watchlist
        sim_matrix = data.get_similarity_matrix(filtered=True)

        candidate_cols = [c for c in sim_matrix.columns if c != "id"]

        # Per-cluster mean similarity to each candidate, with bounded rating modifier
        cluster_sims = (
            sim_matrix.with_columns(pl.exclude("id").fill_nan(0.0))
            .join(compare_df.select("id", "cluster", "score"), how="left", on="id")
            .group_by("cluster", maintain_order=True)
            .agg(
                pl.exclude("cluster", "id", "score").mean(),
                rating=statistics.bounded_rating_modifier(pl.col("score").mean()),
            )
            .with_columns(*((pl.col(col) * pl.col("rating")).alias(col) for col in candidate_cols))
            .drop("rating")
        )

        # For each candidate, pick top-N clusters and aggregate with decay
        cluster_only = cluster_sims.select(pl.exclude("cluster"))

        scores = [
            statistics.weighted_top_k(
                cluster_only[col].sort(descending=True).head(self.TOP_N).to_list(),
                self.DECAY_WEIGHTS,
            )
            for col in candidate_cols
        ]

        score_series = pl.Series(scores)

        # Cluster cohesion confidence
        cohesion = self.compute_cluster_cohesion(data)
        confidence = self.feature_confidence(data.seasonal) * cohesion

        return ScorerResult(statistics.rank_series(score_series), confidence)

    def compute_cluster_cohesion(self, data):
        """Mean intra-cluster similarity as confidence signal."""
        sim_matrix = data.get_similarity_matrix(filtered=False)
        watchlist = data.watchlist

        candidate_cols = [c for c in sim_matrix.columns if c != "id"]
        watchlist_ids = set(sim_matrix["id"].cast(pl.Utf8).to_list())

        intra_cols = [c for c in candidate_cols if c in watchlist_ids]
        if not intra_cols:
            return 1.0

        cluster_series = watchlist.select("id", "cluster")

        long = (
            sim_matrix.select("id", *intra_cols)
            .unpivot(index="id", on=intra_cols, variable_name="col_id", value_name="similarity")
            .filter(pl.col("id").cast(pl.Utf8) != pl.col("col_id"))
            .join(cluster_series, on="id", how="left")
            .join(
                cluster_series.rename({"id": "col_id_int", "cluster": "col_cluster"}),
                left_on=pl.col("col_id").cast(pl.UInt32),
                right_on="col_id_int",
                how="left",
            )
            .filter(pl.col("cluster") == pl.col("col_cluster"))
        )

        if len(long) == 0:
            return 0.5

        mean_cohesion = long["similarity"].mean()

        return min(mean_cohesion / 0.5, 1.0)


class DirectSimilarityScorer(AbstractScorer):
    """Score candidates by structural resemblance to specific watched items.

    Uses top-K matches with geometric decay. Dropped/paused shows contribute
    negative signal. Zero-overlap candidates get zero score and zero confidence.
    """

    name = "directscore"

    TOP_K = 5
    DECAY_WEIGHTS: ClassVar[list[float]] = [1.0, 0.6, 0.36, 0.22, 0.13]
    DROP_PENALTY = 0.5
    MIN_SIMILARITY_THRESHOLD = 0.3

    def score(self, data):
        compare_df = data.watchlist
        user_mean = statistics.mean_score_default(compare_df, 5.0)

        long = self.build_match_table(data, compare_df, user_mean)
        ranked = self.rank_and_weight_matches(long)
        result = self.aggregate_scores(ranked, data)

        score = result["score"].clip(lower_bound=0.0)

        match_conf = (result["best_sim"] / self.MIN_SIMILARITY_THRESHOLD).clip(upper_bound=1.0)
        confidence = self.feature_confidence(data.seasonal) * match_conf

        return ScorerResult(statistics.rank_series(score), confidence)

    def build_match_table(self, data, compare_df, user_mean):
        """Melt similarity matrix to long format with rating modifiers."""
        sim_matrix = data.get_similarity_matrix(filtered=True)
        candidate_cols = [c for c in sim_matrix.columns if c != "id"]

        return (
            sim_matrix.unpivot(
                index="id",
                on=candidate_cols,
                variable_name="candidate_id",
                value_name="similarity",
            )
            .join(compare_df.select("id", "score", "user_status"), on="id", how="left")
            .with_columns(
                similarity=pl.col("similarity").fill_nan(None),
                rating_mod=pl.when(pl.col("user_status").is_in(["DROPPED", "PAUSED"]))
                .then(-self.DROP_PENALTY)
                .otherwise(statistics.bounded_rating_modifier(pl.col("score"), user_mean)),
            )
            .with_columns(
                match_score=pl.col("similarity").fill_null(0.0) * pl.col("rating_mod"),
            )
        )

    def rank_and_weight_matches(self, long):
        """Rank matches per candidate and apply geometric decay weights via join."""
        weights_df = pl.DataFrame(
            {
                "rank": list(range(1, self.TOP_K + 1)),
                "weight": self.DECAY_WEIGHTS[: self.TOP_K],
            }
        ).cast({"rank": pl.UInt32})

        return (
            long.with_columns(
                rank=pl.col("similarity")
                .rank(method="ordinal", descending=True)
                .over("candidate_id")
                .cast(pl.UInt32)
            )
            .filter(pl.col("rank") <= self.TOP_K)
            .join(weights_df, on="rank", how="left")
        )

    def aggregate_scores(self, ranked, data):
        """Aggregate weighted match scores per candidate, preserving original order."""
        sim_matrix = data.get_similarity_matrix(filtered=True)
        candidate_cols = [c for c in sim_matrix.columns if c != "id"]

        scores_df = ranked.group_by("candidate_id", maintain_order=True).agg(
            score=(pl.col("match_score") * pl.col("weight")).sum() / pl.col("weight").sum(),
            best_sim=pl.col("similarity").max(),
        )

        order = pl.DataFrame({"candidate_id": candidate_cols})
        return order.join(scores_df, on="candidate_id", how="left").with_columns(
            pl.col("score").fill_null(0.0),
            pl.col("best_sim").fill_null(0.0),
        )


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

        rated_confidence = (popularity.log1p() / pl.lit(self.MIN_MEMBER_COUNT).log1p()).clip(
            upper_bound=1.0
        )

        result = scoring_target_df.select(
            score=pl.when(has_score).then(rated_score).otherwise(hype_score),
            confidence=pl.when(has_score)
            .then(rated_confidence)
            .otherwise(pl.lit(self.HYPE_CONFIDENCE)),
        )

        return ScorerResult(result["score"], result["confidence"])


class ContinuationScorer(AbstractScorer):
    name = "continuationscore"

    DROP_CAP = 0.15
    SUMMARY_FACTOR = 0.3

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
                completion_weight=pl.col("user_status")
                .replace_strict(
                    {"COMPLETED": 1.0, "CURRENT": 0.7, "PAUSED": 0.3, "DROPPED": 0.1},
                    default=0.0,
                )
                .cast(pl.Float64),
                was_dropped=pl.col("user_status") == "DROPPED",
            )
            .with_columns(
                strength=pl.col("predecessor_rating") * pl.col("completion_weight"),
            )
            .group_by("id", maintain_order=True)
            .agg(
                pl.col("strength").max(),
                # A dropped sequel overrides earlier positive signal
                any_dropped=pl.col("was_dropped").any(),
            )
            .with_columns(
                strength=pl.when(pl.col("any_dropped"))
                .then(pl.col("strength").clip(upper_bound=self.DROP_CAP))
                .otherwise(pl.col("strength"))
            )
            .drop("any_dropped")
        )

        score = joined["strength"].fill_null(0.0)

        if "is_summary" in scoring_target_df.columns:
            summary_mask = scoring_target_df["is_summary"]
            score = pl.select(
                pl.when(summary_mask).then(score * self.SUMMARY_FACTOR).otherwise(score)
            ).to_series()

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


class CollaborativeRecommendationScorer(AbstractScorer):
    """Score candidates using AniList community recommendation links.

    For each watchlist item that has recommendation links pointing to a seasonal
    candidate, compute a signal based on normalized thumbs, user rating, and
    watch status. Sum signals per candidate — more links means more evidence.
    """

    name = "collaborativescore"

    MIN_LINK_COUNT = 3

    STATUS_MODIFIERS: ClassVar[dict[str, float]] = {
        "COMPLETED": 1.0,
        "REPEATING": 1.0,
        "CURRENT": 0.8,
        "PAUSED": 0.3,
        "DROPPED": -0.5,
    }

    def score(self, data):
        watchlist = data.watchlist
        seasonal = data.seasonal
        n = len(seasonal)

        if watchlist is None or "recommendations" not in watchlist.columns:
            return ScorerResult(pl.Series([0.0] * n), pl.Series([0.0] * n))

        user_mean = statistics.mean_score_default(watchlist, 5.0)
        links = self._build_link_table(watchlist, seasonal, user_mean)

        if len(links) == 0:
            return ScorerResult(pl.Series([0.0] * n), pl.Series([0.0] * n))

        aggregated = self._aggregate_signals(links)
        result = self._join_to_seasonal(aggregated, seasonal)

        return ScorerResult(statistics.rank_series(result["signal"]), result["confidence"])

    def _build_link_table(self, watchlist, seasonal, user_mean):
        """Explode recommendations, filter to seasonal candidates, compute per-link signal."""
        seasonal_ids = seasonal["id"].to_list()

        # Sum of absolute ratings per watchlist item for normalization
        wl = watchlist.with_columns(
            total_thumbs=pl.col("recommendations")
            .list.eval(pl.element().struct.field("rating").abs())
            .list.sum()
            .cast(pl.Float64)
        )

        exploded = (
            wl.select("id", "score", "user_status", "total_thumbs", "recommendations")
            .explode("recommendations")
            .filter(pl.col("recommendations").is_not_null())
            .unnest("recommendations")
            .filter(pl.col("recommended_id").is_in(seasonal_ids))
        )

        if len(exploded) == 0:
            return exploded

        return exploded.with_columns(
            normalized_thumbs=pl.when(pl.col("total_thumbs") > 0)
            .then(pl.col("rating").cast(pl.Float64) / pl.col("total_thumbs"))
            .otherwise(0.0),
            status_modifier=pl.col("user_status")
            .replace_strict(self.STATUS_MODIFIERS, default=0.0)
            .cast(pl.Float64),
            rating_modifier=statistics.bounded_rating_modifier(pl.col("score"), user_mean),
        ).with_columns(
            link_signal=pl.when(pl.col("user_status") == "DROPPED")
            .then(pl.col("normalized_thumbs") * pl.col("status_modifier"))
            .otherwise(
                pl.col("normalized_thumbs") * pl.col("rating_modifier") * pl.col("status_modifier")
            ),
        )

    def _aggregate_signals(self, links):
        """Group by candidate, sum signals, compute confidence.

        Confidence is purely link-count based — thumb strength already
        modulates the signal itself via normalized_thumbs.
        """
        return (
            links.group_by("recommended_id", maintain_order=True)
            .agg(
                signal=pl.col("link_signal").sum(),
                link_count=pl.len(),
            )
            .with_columns(
                confidence=(pl.col("link_count").cast(pl.Float64) / self.MIN_LINK_COUNT).clip(
                    upper_bound=1.0
                ),
            )
        )

    def _join_to_seasonal(self, aggregated, seasonal):
        """Join aggregated scores back to seasonal ordering."""
        return (
            seasonal.select("id")
            .join(
                aggregated.select(
                    pl.col("recommended_id").alias("id"),
                    "signal",
                    "confidence",
                ),
                on="id",
                how="left",
            )
            .with_columns(
                pl.col("signal").fill_null(0.0),
                pl.col("confidence").fill_null(0.0),
            )
        )
