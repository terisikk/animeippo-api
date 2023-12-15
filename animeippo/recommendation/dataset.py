from functools import lru_cache


class UserDataSet:
    """Collection of dataframes and other data related to
    the recommendation system."""

    def __init__(self, watchlist, seasonal, features=None):
        self.watchlist = watchlist
        self.seasonal = seasonal
        self.mangalist = None
        self.recommendations = None
        self.all_features = features
        self.nsfw_tags = []
        self.user_favourite_genres = None

    @lru_cache(maxsize=5)
    def watchlist_explode_cached(self, column):
        return self.watchlist.explode(column)

    @lru_cache(maxsize=5)
    def recommendations_explode_cached(self, column):
        return self.recommendations.explode(column)
