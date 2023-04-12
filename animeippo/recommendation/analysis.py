import pandas as pd
import numpy as np
import scipy.spatial.distance as scdistance


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


def mean_score_for_categorical_values(dataframe, field):
    gdf = dataframe.explode(field)

    return gdf.groupby(field)["score"].mean()


def weight_genres_by_user_score(categoricals, averages):
    mean = np.nanmean([averages.get(categorical, np.nan) for categorical in categoricals])
    mean = mean if not np.isnan(mean) else 0.0

    return mean


def weight_studios_by_user_score(categoricals, averages):
    mode = averages.mode()[0]

    mean = np.nanmean([averages.get(categorical, mode) for categorical in categoricals])
    mean = mean if not np.isnan(mean) else 0.0

    return mean


def fill_status_data_from_user_list(dataframe, user_dataframe):
    dataframe["status"] = np.nan
    dataframe["status"].update(user_dataframe["status"])
