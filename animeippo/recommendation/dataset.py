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
