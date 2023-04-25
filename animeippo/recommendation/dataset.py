import numpy as np


class UserDataSet:
    def __init__(self, watchlist, seasonal):
        self.watchlist = watchlist
        self.seasonal = seasonal
        self.recommendations = None

        if self.seasonal is not None and self.watchlist is not None:
            self.fill_status_data_from_watchlist()

    def fill_status_data_from_watchlist(self):
        self.seasonal["status"] = np.nan
        self.seasonal["status"].update(self.watchlist["status"])
