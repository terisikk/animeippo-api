import polars as pl
import structlog

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
        dataset.fit(self.encoder, self.clustering_model)

        recommendations = self.score_anime(dataset)

        predictions = self.clustering_model.predict(
            dataset.seasonal["encoded"], dataset.get_similarity_matrix(filtered=False)
        )
        recommendations = recommendations.with_columns(
            cluster=predictions["cluster"].cast(pl.UInt32),
        )

        return recommendations.sort("discovery_score", descending=True)

    def score_anime(self, dataset):
        scoring_target_df = dataset.seasonal
        n = len(scoring_target_df)

        if not self.discovery_scorers:
            raise RuntimeError("No scorers added for engine. Please add at least one.")

        # Run discovery scorers and collect results
        discovery_results = {}
        for scorer in self.discovery_scorers:
            discovery_results[scorer.name] = (
                self.run_scorer(scorer, dataset, n),
                scorer.weight,
            )

        # Run engagement scorers separately
        engagement_results = {}
        for scorer in self.engagement_scorers:
            engagement_results[scorer.name] = self.run_scorer(scorer, dataset, n)

        # Store confidence-adjusted scores so categories sorting by individual
        # scorers respect data quality (e.g. unknown studio = low confidence = low score)
        scoring_target_df = scoring_target_df.with_columns(
            **{
                name: result.score * result.confidence
                for name, (result, _) in discovery_results.items()
            },
            **{
                name: result.score * result.confidence
                for name, result in engagement_results.items()
            },
        )

        # Confidence-weighted blending for discovery score
        effective_weight_cols = []
        weighted_score_cols = []
        conf_cols = {}

        for name, (result, base_weight) in discovery_results.items():
            ew = (result.confidence * base_weight).alias(f"_ew_{name}")
            effective_weight_cols.append(ew)
            weighted_score_cols.append(
                (result.score * result.confidence * base_weight).alias(f"_ws_{name}")
            )
            conf_cols[f"{name}_conf"] = result.confidence

        ew_names = [f"_ew_{name}" for name in discovery_results]
        ws_names = [f"_ws_{name}" for name in discovery_results]

        scoring_target_df = scoring_target_df.with_columns(
            *effective_weight_cols,
            *weighted_score_cols,
            **conf_cols,
        )

        total_weight = pl.sum_horizontal(*ew_names)
        total_base_weight = sum(w for _, w in discovery_results.values())

        overall_confidence = (
            pl.sum_horizontal(
                *(pl.col(f"{name}_conf") * w for name, (_, w) in discovery_results.items())
            )
            / total_base_weight
        )

        # Fallback uses uniform weights when all confidences are zero
        raw_discovery = (
            pl.when(total_weight > 0)
            .then(pl.sum_horizontal(*ws_names) / total_weight)
            .otherwise(0.0)
        )

        scoring_target_df = scoring_target_df.with_columns(
            discovery_score=raw_discovery,
            overall_confidence=overall_confidence,
        )

        # Add engagement columns
        for name, result in engagement_results.items():
            scoring_target_df = scoring_target_df.with_columns(
                **{f"{name}_confidence": result.confidence}
            )

        # Clean up temporary columns
        scoring_target_df = scoring_target_df.drop(
            [c for c in scoring_target_df.columns if c.startswith("_ew_") or c.startswith("_ws_")]
        )

        return scoring_target_df

    def run_scorer(self, scorer, dataset, n=0):
        try:
            return scorer.score(dataset)
        except Exception:
            logger.error("scorer_error", scorer=scorer.name, exc_info=True)
            return ScorerResult(
                score=pl.Series([0.0] * n),
                confidence=pl.Series([0.0] * n),
            )

    def categorize_anime(self, data):
        if self.ranking_orchestrator is None:
            raise RuntimeError("No ranking orchestrator configured for engine.")
        return self.ranking_orchestrator.render(data)

    def add_scorer(self, scorer):
        self.discovery_scorers.append(scorer)
