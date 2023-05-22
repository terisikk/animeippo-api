import numpy as np
import pandas as pd


class UserDataSet:
    def __init__(self, watchlist, seasonal, features=None):
        self.watchlist = watchlist
        self.seasonal = seasonal
        self.features = features if features else []
        self.recommendations = None

        if self.seasonal is not None and self.watchlist is not None:
            self.seasonal = fill_status_data_from_watchlist(self.seasonal, self.watchlist)

        if self.seasonal is not None:
            self.seasonal["features"] = fill_feature_data(self.seasonal, self.features)

        if self.watchlist is not None:
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

        if isinstance(value, list) or isinstance(value, np.ndarray):
            all_features.extend([v for v in value if not pd.isnull(v)])
        elif value is None or pd.isnull(value):
            continue
        else:
            all_features.append(value)

    return all_features
