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

    def __init__(
        self,
        clustering_model,
        encoder,
        discovery_scorers=None,
        engagement_scorers=None,
        ranking_orchestrator=None,
    ):
        self.clustering_model = clustering_model
        self.encoder = encoder
        self.discovery_scorers = discovery_scorers or []
        self.engagement_scorers = engagement_scorers or []
        self.ranking_orchestrator = ranking_orchestrator

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
