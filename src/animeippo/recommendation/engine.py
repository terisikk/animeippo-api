import polars as pl


class AnimeRecommendationEngine:
    """Generates recommendations based on given scorers and categorizers. Optionally accepts
    a custom clustering model and feature encoder."""

    def __init__(self, clustering_model, encoder, scorers=None, categorizers=None):
        self.clustering_model = clustering_model
        self.encoder = encoder
        self.scorers = scorers or []
        self.categorizers = categorizers or []

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
                **{scorer.name: self.run_scorer(scorer, dataset) for scorer in self.scorers}
            )

            names = [scorer.name for scorer in self.scorers]
            mean_score = scoring_target_df.select(names).mean_horizontal()

            scoring_target_df = scoring_target_df.with_columns(
                recommend_score=mean_score,
                final_score=mean_score,
                discourage_score=1.0,
            )

        else:
            raise RuntimeError("No scorers added for engine. Please add at least one.")

        return scoring_target_df

    def run_scorer(self, scorer, dataset):
        try:
            return scorer.score(dataset).fill_nan(0.0)
        except Exception as e:
            print(f"Error in scorer {scorer.name}: {e}")
            return 0.0

    def categorize_anime(self, data):
        cats = []

        for grouper in self.categorizers:
            group = grouper.categorize(data)

            if group is not None and len(group) > 0:
                items = group["id"].to_list()
                cats.append({"name": grouper.description, "items": items})

        return cats

    def add_scorer(self, scorer):
        self.scorers.append(scorer)

    def add_categorizer(self, categorizer):
        self.categorizers.append(categorizer)
