import copy

import polars as pl
import structlog

from .funnel import add_funnel_metadata
from .scoring import ScorerResult

logger = structlog.get_logger()


class AnimeRecommendationEngine:
    """Generates recommendations based on given scorers and ranking orchestrator.

    Optionally accepts a custom clustering model and feature encoder.
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        clustering_model,
        encoder,
        discovery_scorers=None,
        engagement_scorers=None,
        ranking_orchestrator=None,
        cluster_namer=None,
    ):
        self.clustering_model = clustering_model
        self.encoder = encoder
        self.discovery_scorers = discovery_scorers or []
        self.engagement_scorers = engagement_scorers or []
        self.ranking_orchestrator = ranking_orchestrator
        self.cluster_namer = cluster_namer

        self._cluster_names = None

    def fit_predict(self, dataset):
        # Fresh copies so concurrent requests don't share mutable fit state
        encoder = copy.copy(self.encoder)
        clustering_model = copy.copy(self.clustering_model)

        dataset.fit(encoder, clustering_model)

        recommendations = self.score_anime(dataset)

        predictions = clustering_model.predict(
            dataset.seasonal["encoded"], dataset.get_similarity_matrix(filtered=False)
        )
        recommendations = recommendations.with_columns(
            cluster=predictions["cluster"].cast(pl.UInt32),
            cluster_similarity=predictions["similarity"],
        )

        recommendations = add_funnel_metadata(recommendations)

        return recommendations.sort("discovery_score", descending=True)

    def score_anime(self, dataset):
        if not self.discovery_scorers:
            raise RuntimeError("No scorers added for engine. Please add at least one.")

        scoring_target_df = dataset.seasonal
        n = len(scoring_target_df)
        discovery_results = self.calculate_scores(dataset, n, self.discovery_scorers)
        engagement_results = self.calculate_scores(dataset, n, self.engagement_scorers)

        # Store confidence-adjusted scores so categories sorting by individual
        # scorers respect data quality (e.g. unknown studio = low confidence = low score)
        scoring_target_df = scoring_target_df.with_columns(
            **{result.name: result.score * result.confidence for result in discovery_results},
            **{result.name: result.score * result.confidence for result in engagement_results},
        )

        scoring_target_df = scoring_target_df.with_columns(
            discovery_score=self.calculate_discovery_score(discovery_results),
        )

        # Add engagement columns
        for result in engagement_results:
            scoring_target_df = scoring_target_df.with_columns(
                **{f"{result.name}_confidence": result.confidence}
            )

        return scoring_target_df

    def calculate_discovery_score(self, discovery_results):
        # Confidence-weighted blending for discovery score
        # ew = conf * base_weight
        # ws = conf * base_weight * score
        # tw = sum(ew) = sum(conf * base_weight)
        # ds = sum(ws) / tw = sum(ws) / sum(ew)
        #    = sum(conf * base_weight * score) / sum(conf * base_weight)

        effective_weights = []
        weighted_scores = []

        for result in discovery_results:
            ew = result.confidence * result.weight
            effective_weights.append(ew)
            weighted_scores.append(result.score * ew)

        total_effective_weight = pl.sum_horizontal(effective_weights)
        total_weighted_score = pl.sum_horizontal(weighted_scores)

        # Fallback uses uniform weights when all confidences are zero
        return (
            pl.when(total_effective_weight > 0)
            .then(total_weighted_score / total_effective_weight)
            .otherwise(0.0)
        )

    def calculate_scores(self, dataset, n, scorers):
        return [self.run_scorer(scorer, dataset, n) for scorer in scorers]

    def run_scorer(self, scorer, dataset, n=0):
        try:
            return scorer.score(dataset)
        except Exception:
            logger.exception("scorer_error", scorer=scorer.name)
            return ScorerResult(
                name=scorer.name,
                score=pl.Series([0.0] * n),
                confidence=pl.Series([0.0] * n),
                weight=scorer.weight,
            )

    def categorize_anime(self, data):
        if self.ranking_orchestrator is None:
            raise RuntimeError("No ranking orchestrator configured for engine.")
        return self.ranking_orchestrator.render(data)

    def add_scorer(self, scorer):
        self.discovery_scorers.append(scorer)

    def get_clusters(self, dataset):
        # TODO: We might need to calculate these even if either is missing
        if (
            "cluster" not in dataset.watchlist.columns
            or "cluster" not in dataset.recommendations.columns
        ):
            return pl.DataFrame()

        names = self.get_cluster_names(dataset)
        stats = self.get_cluster_stats(dataset)

        return names.join(stats, on="cluster", how="left")

    def get_cluster_names(self, dataset):
        feature_df = dataset.watchlist.explode("features", empty_as_null=True)
        feature_df = feature_df.filter(~pl.col("features").is_in(dataset.nsfw_tags))

        cluster_names = self.cluster_namer.name_clusters_from_data(
            feature_df, "features", "cluster"
        )

        # TODO: Kinda stupid to convert back to dataframe here but
        # it's a temporary measure until we change cluster namer to
        # work on dataframes too, which it probably should
        return pl.DataFrame(
            {
                "cluster": [int(cluster) for cluster in cluster_names.keys()],
                "name": cluster_names.values(),
            }
        )

    def get_cluster_stats(self, dataset):
        """
        Calculcate statistics for clusters.
        Rank clusters by user enjoyment * content volume * match quality.
        """
        watchlist = dataset.watchlist
        recommendations = dataset.recommendations

        user_scores = (
            watchlist.filter(pl.col("cluster") >= 0)
            .group_by("cluster")
            .agg(
                mean_score=pl.col("score").mean(),
                count=pl.col("id").len(),
                completed_count=(pl.col("user_status") == "COMPLETED").sum(),
            )
            .with_columns(
                completion_rate=(pl.col("completed_count") / pl.col("count") * 100).round(1),
                mean_score=pl.col("mean_score").round(1),
            )
        )

        rec_stats = (
            recommendations.filter(pl.col("cluster").is_not_null())
            .group_by("cluster")
            .agg(
                rec_count=pl.col("id").len(),
                mean_similarity=pl.col("cluster_similarity").mean(),
            )
        )

        return user_scores.join(rec_stats, on="cluster", how="left").with_columns(
            rank_score=pl.col("mean_score")
            * pl.col("rec_count").log1p()
            * pl.col("mean_similarity")
        )
