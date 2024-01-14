import polars as pl
from animeippo.clustering import model, metrics

from animeippo.recommendation import encoding


class AnimeRecommendationEngine:
    """Generates recommendations based on given scorers and categorizers. Optionally accepts
    a custom clustering model and feature encoder."""

    def __init__(self, scorers=None, categorizers=None, clustering_model=None, encoder=None):
        self.scorers = scorers or []
        self.categorizers = categorizers or []
        self.clustering_model = clustering_model or model.AnimeClustering()
        self.encoder = encoder or encoding.CategoricalEncoder()

    def validate(self, dataset):
        is_missing_seasonal = dataset.seasonal is None
        is_missing_watchlist = dataset.watchlist is None

        if is_missing_seasonal or is_missing_watchlist:
            error_desc = (
                f"Watchlist invalid?: {is_missing_watchlist}. "
                + f"Seasonal invalid?: {is_missing_seasonal}"
            )

            raise RuntimeError("Trying to recommend anime without proper data. " + error_desc)

    def fit(self, dataset):
        self.validate(dataset)

        dataset.all_features = self.extract_features(dataset)

        self.encoder.fit(dataset.all_features)

        dataset.watchlist = dataset.watchlist.with_columns(
            encoded=self.encoder.encode(dataset.watchlist)
        )
        dataset.seasonal = dataset.seasonal.with_columns(
            encoded=self.encoder.encode(dataset.seasonal)
        )

        dataset.watchlist = dataset.watchlist.with_columns(
            cluster=self.clustering_model.cluster_by_features(dataset.watchlist)
        )

        dataset.similarity_matrix = metrics.categorical_similarity(
            dataset.watchlist["encoded"],
            dataset.seasonal["encoded"],
            self.clustering_model.distance_metric,
            dataset.seasonal["id"].cast(pl.Utf8),
        ).with_columns(id=dataset.watchlist["id"])
        # Categories could use unfiltered watchlist, but scoring needs to filter it

        # Rechunk to maximize performance, not sure if it has any real effect
        dataset.seasonal = dataset.seasonal.rechunk()
        dataset.watchlist = dataset.watchlist.rechunk()

        return dataset

    def extract_features(self, dataset):
        all_features = pl.concat([dataset.seasonal["features"], dataset.watchlist["features"]])
        all_features = (
            pl.concat([all_features, dataset.all_features])
            if dataset.all_features is not None
            else all_features
        )

        return all_features.explode().unique().drop_nulls()

    def fit_predict(self, dataset):
        dataset = self.fit(dataset)

        recommendations = self.score_anime(dataset)

        recommendations = recommendations.with_columns(
            cluster=self.clustering_model.predict(
                dataset.seasonal["encoded"], dataset.get_similarity_matrix(filtered=False)
            ),
        )

        return recommendations.sort("recommend_score", descending=True)

    def score_anime(self, dataset):
        scoring_target_df = dataset.seasonal

        if len(self.scorers) > 0:
            scoring_target_df = scoring_target_df.with_columns(
                **{scorer.name: scorer.score(dataset).fill_nan(0.0) for scorer in self.scorers}
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
