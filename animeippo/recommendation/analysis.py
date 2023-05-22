import pandas as pd
import numpy as np
import scipy.spatial.distance as scdistance
import sklearn.preprocessing as skpre


def similarity(x_orig, y_orig, metric="jaccard"):
    distances = scdistance.cdist(x_orig, y_orig, metric=metric)

    return 1 - distances


def similarity_of_anime_lists(features1, features2, encoder):
    similarities = pd.DataFrame(
        similarity(encoder.encode(features1.values), encoder.encode(features2.values)),
        index=features1.index,
    )

    return similarities.mean(axis=1, skipna=True)

    # return similarities.apply(np.nanmean, axis=1)


def mean_score_per_categorical(dataframe, column):
    return dataframe.groupby(column)["score"].mean()


def weighted_mean_for_categorical_values(categoricals, weights, fillna=0.0):
    return np.nanmean([weights.get(categorical, fillna) for categorical in categoricals])


def weight_categoricals(dataframe, column):
    exploded = dataframe.explode(column)
    averages = mean_score_per_categorical(exploded, column)

    weights = np.sqrt(exploded[column].value_counts())
    weights = weights * averages

    return weights


def weight_categoricals_z_score(dataframe, column):
    df = dataframe.explode(column)

    scores = df.groupby(column)["score"].mean()
    counts = pd.DataFrame(df[column].value_counts())

    mean = np.mean(scores)
    std = np.std(scores)

    weighted_scores = counts.apply(lambda row: ((scores[row.name] - mean) / std) * row, axis=1)

    return normalize_column(weighted_scores)


def normalize_column(df_column):
    shaped = df_column.to_numpy().reshape(-1, 1)
    return pd.DataFrame(skpre.MinMaxScaler().fit_transform(shaped), index=df_column.index)
