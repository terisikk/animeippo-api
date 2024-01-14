from functools import lru_cache
import polars as pl

from animeippo.clustering import metrics


class RecommendationModel:
    """Collection of dataframes and other data related to
    the recommendation system."""

    def __init__(self, user_profile, seasonal, features=None):
        self.user_profile = user_profile
        self.watchlist = user_profile.watchlist if user_profile is not None else None
        self.seasonal = seasonal
        self.mangalist = user_profile.mangalist if user_profile is not None else None
        self.recommendations = None

        self.all_features = features
        self.nsfw_tags = []
        self.similarity_matrix = None

    def validate(self):
        is_missing_seasonal = self.seasonal is None
        is_missing_watchlist = self.watchlist is None

        if is_missing_seasonal or is_missing_watchlist:
            error_desc = (
                f"Watchlist invalid?: {is_missing_watchlist}. "
                + f"Seasonal invalid?: {is_missing_seasonal}"
            )

            raise RuntimeError("Trying to recommend anime without proper data. " + error_desc)

    def encode(self, encoder):
        self.all_features = self.extract_features(self)

        encoder.fit(self.all_features)

        self.watchlist = self.watchlist.with_columns(encoded=encoder.encode(self.watchlist))
        self.seasonal = self.seasonal.with_columns(encoded=encoder.encode(self.seasonal))

    def fit(self, encoder, clustering_model):
        self.validate()
        self.encode(encoder)

        self.watchlist = self.watchlist.with_columns(
            cluster=clustering_model.cluster_by_features(self.watchlist)
        )

        self.similarity_matrix = metrics.categorical_similarity(
            self.watchlist["encoded"],
            self.seasonal["encoded"],
            clustering_model.distance_metric,
            self.seasonal["id"].cast(pl.Utf8),
        ).with_columns(id=self.watchlist["id"])
        # Categories could use unfiltered watchlist, but scoring needs to filter it

        # Rechunk to maximize performance, not sure if it has any real effect
        self.seasonal = self.seasonal.rechunk()
        self.watchlist = self.watchlist.rechunk()

    def extract_features(self, dataset):
        all_features = pl.concat([dataset.seasonal["features"], dataset.watchlist["features"]])
        all_features = (
            pl.concat([all_features, dataset.all_features])
            if dataset.all_features is not None
            else all_features
        )

        return all_features.explode().unique().drop_nulls()

    @lru_cache(maxsize=10)
    def watchlist_explode_cached(self, column):
        return self.watchlist.explode(column)

    @lru_cache(maxsize=10)
    def recommendations_explode_cached(self, column):
        return self.recommendations.explode(column)

    @lru_cache(maxsize=10)
    def seasonal_explode_cached(self, column):
        return self.seasonal.explode(column)

    @lru_cache(maxsize=10)
    def get_similarity_matrix(self, filtered=False):
        if filtered:
            return self.similarity_matrix.filter(~pl.col("id").is_in(self.seasonal["id"]))

        return self.similarity_matrix
