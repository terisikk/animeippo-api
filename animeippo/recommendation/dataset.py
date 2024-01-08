from functools import lru_cache


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

    @lru_cache(maxsize=10)
    def watchlist_explode_cached(self, column):
        return self.watchlist.explode(column)

    @lru_cache(maxsize=10)
    def recommendations_explode_cached(self, column):
        return self.recommendations.explode(column)

    @lru_cache(maxsize=10)
    def seasonal_explode_cached(self, column):
        return self.seasonal.explode(column)
