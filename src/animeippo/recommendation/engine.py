import logging

import polars as pl

logger = logging.getLogger(__name__)


class AnimeRecommendationEngine:
    """Generates recommendations based on given scorers and ranking orchestrator.

    Optionally accepts a custom clustering model and feature encoder.
    """

    def __init__(self, clustering_model, encoder, scorers=None, ranking_orchestrator=None):
        self.clustering_model = clustering_model
        self.encoder = encoder
        self.scorers = scorers or []
        self.ranking_orchestrator = ranking_orchestrator

    def fit_predict(self, dataset):
        dataset.fit(self.encoder, self.clustering_model)

        recommendations = self.score_anime(dataset)

        recommendations = recommendations.with_columns(
            cluster=self.clustering_model.predict(
                dataset.seasonal["encoded"], dataset.get_similarity_matrix(filtered=False)
            ).cast(pl.UInt32),
        )

        return recommendations.sort("recommend_score", descending=True)

    def score_anime(self, dataset):
        scoring_target_df = dataset.seasonal

        if len(self.scorers) > 0:
            scoring_target_df = scoring_target_df.with_columns(
                **{
                    scorer.name: self.run_scorer(scorer, dataset) * getattr(scorer, "weight", 1.0)
                    for scorer in self.scorers
                }
            )

            recommend_score = sum(scoring_target_df[scorer.name] for scorer in self.scorers)

            scoring_target_df = scoring_target_df.with_columns(
                recommend_score=recommend_score,
                adjusted_score=recommend_score,
                discourage_score=1.0,
            )

        else:
            raise RuntimeError("No scorers added for engine. Please add at least one.")

        return scoring_target_df

    def run_scorer(self, scorer, dataset):
        try:
            return scorer.score(dataset).fill_nan(0.0)
        except Exception as e:
            logger.exception(f"Error in scorer {scorer.name}: {e}")
            return 0.0

    def categorize_anime(self, data):
        if self.ranking_orchestrator is None:
            raise RuntimeError("No ranking orchestrator configured for engine.")
        return self.ranking_orchestrator.render(data)

    def add_scorer(self, scorer):
        self.scorers.append(scorer)
