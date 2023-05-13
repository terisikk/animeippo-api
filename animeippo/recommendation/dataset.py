import numpy as np


class UserDataSet:
    def __init__(self, watchlist, seasonal, features=None):
        self.watchlist = watchlist
        self.seasonal = seasonal
        self.features = features if features else []
        self.recommendations = None

        if self.seasonal is not None and self.watchlist is not None:
            self.seasonal = fill_status_data_from_watchlist(self.seasonal, self.watchlist)
            self.seasonal["features"] = fill_feature_data(self.seasonal, self.features)
            self.watchlist["features"] = fill_feature_data(self.watchlist, self.features)


def fill_status_data_from_watchlist(seasonal, watchlist):
    seasonal["status"] = np.nan
    seasonal["status"].update(watchlist["status"])
    return seasonal


def fill_feature_data(dataframe, features):
    return dataframe.apply(get_features, args=(features,), axis=1)


def get_features(row, feature_names):
    all_features = []

    for feature in feature_names:
        value = row[feature]

        if isinstance(value, str):
            all_features.append(value)
        elif isinstance(value, list):
            all_features.extend(value)
        else:
            continue

    return all_features
