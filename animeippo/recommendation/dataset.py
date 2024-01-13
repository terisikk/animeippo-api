from functools import lru_cache
import polars as pl


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

    def fit():
        pass

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
