class UserDataSet:
    def __init__(self, watchlist, seasonal, features=None):
        self.watchlist = watchlist
        self.seasonal = seasonal
        self.recommendations = None
        self.all_features = None
