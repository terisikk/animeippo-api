import pandas as pd
import numpy as np
import scipy.spatial.distance as scdistance
import sklearn.preprocessing as skpre


def similarity(x_orig, y_orig, metric="jaccard"):
    distances = scdistance.cdist(x_orig, y_orig, metric=metric)

    return 1 - distances


def similarity_of_anime_lists(dataframe1, dataframe2, encoder):
    similarities = pd.DataFrame(
        similarity(encoder.encode(dataframe1["genres"]), encoder.encode(dataframe2["genres"])),
        index=dataframe1.index,
    )
    similarities = similarities.apply(np.nanmean, axis=1)

    return similarities


def mean_score_per_categorical(dataframe, column):
    gdf = dataframe.explode(column)

    return gdf.groupby(column)["score"].mean()


def weighted_mean_for_categorical_values(categoricals, averages, fillna=0.0):
    averages = averages.fillna(fillna)
    return np.nanmean([averages.get(categorical, fillna) for categorical in categoricals])


def weight_categoricals(dataframe, column):
    averages = mean_score_per_categorical(dataframe, column)
    averages = averages / 10

    weights = np.sqrt(dataframe.explode(column)[column].value_counts())

    weights = weights * averages

    return weights


def fill_status_data_from_user_list(dataframe, user_dataframe):
    dataframe["status"] = np.nan
    dataframe["status"].update(user_dataframe["status"])


def normalize_column(df_column):
    shaped = df_column.to_numpy().reshape(-1, 1)
    return pd.DataFrame(skpre.MinMaxScaler().fit_transform(shaped), index=df_column.index)
