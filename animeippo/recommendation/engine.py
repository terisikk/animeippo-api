import numpy as np
import pandas as pd
import polars as pl

from animeippo.recommendation import encoding, clustering, analysis


class AnimeRecommendationEngine:
    """Generates recommendations based on given scorers and categorizers. Optionally accepts
    a custom clustering model and feature encoder."""

    def __init__(self, scorers=None, categorizers=None, clustering_model=None, encoder=None):
        self.scorers = scorers or []
        self.categorizers = categorizers or []
        self.clustering_model = clustering_model or clustering.AnimeClustering()
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

        all_features = pl.concat([dataset.seasonal["features"], dataset.watchlist["features"]])
        all_features = (
            pl.concat([all_features, dataset.all_features])
            if dataset.all_features is not None
            else all_features
        )

        dataset.all_features = all_features.explode().unique().drop_nulls()

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

        filtered_watchlist = dataset.watchlist.filter(~pl.col("id").is_in(dataset.seasonal["id"]))

        dataset.similarity_matrix = analysis.categorical_similarity(
            filtered_watchlist["encoded"],
            dataset.seasonal["encoded"],
            self.clustering_model.distance_metric,
            dataset.seasonal["id"].cast(pl.Utf8),
        ).with_columns(id=filtered_watchlist["id"])
        # Categories could use unfiltered watchlist, but scoring needs to filter it

        # Rechunk to maximize performance, not sure if it has any real effect
        dataset.seasonal = dataset.seasonal.rechunk()
        dataset.watchlist = dataset.watchlist.rechunk()

        return dataset

    def fit_predict(self, dataset):
        dataset = self.fit(dataset)

        recommendations = self.score_anime(dataset)

        recommendations = recommendations.with_columns(
            cluster=self.clustering_model.predict(
                dataset.seasonal["encoded"], dataset.similarity_matrix
            ),
        )

        return recommendations.sort("recommend_score", descending=True)

    def score_anime(self, dataset):
        scoring_target_df = dataset.seasonal

        if len(self.scorers) > 0:
            names = []

            for scorer in self.scorers:
                scoring = scorer.score(dataset)
                scoring_target_df = scoring_target_df.with_columns(
                    **{scorer.name: np.nan_to_num(scoring, nan=0.0)}
                )
                names.append(scorer.name)

            scoring_target_df = scoring_target_df.with_columns(
                recommend_score=scoring_target_df.select(names).mean_horizontal(),
                final_score=scoring_target_df.select(names).mean_horizontal(),
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
