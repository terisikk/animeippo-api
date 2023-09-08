import pandas as pd
import numpy as np
import scipy.spatial.distance as scdistance
import sklearn.preprocessing as skpre


def distance(x_orig, y_orig, metric="jaccard"):
    """
    Calculate pairwise distance between two series.
    Just a wrapper for scipy cdist for a matching
    signature with analysis.similairty."""
    return scdistance.cdist(x_orig, y_orig, metric=metric)


def similarity(x_orig, y_orig, metric="jaccard"):
    """Calculate similarity between two series."""
    distances = distance(x_orig, y_orig, metric=metric)
    return 1 - distances


def categorical_similarity(features1, features2, index=None, metric="jaccard"):
    """Calculate similarity between two series of categorical features. Assumes a series
    that contains vector-encoded representation of features. Discards similarities
    with identical id:s by setting them to nan."""
    if index is None:
        index = features1.index

    similarities = pd.DataFrame(
        similarity(np.vstack(features1), np.vstack(features2), metric=metric),
        index=index,
        columns=features2.index,
    )

    similarities = discard_similarities_with_self(similarities)

    return similarities


def discard_similarities_with_self(dataframe):
    for column in dataframe.columns:
        dataframe.loc[column, column] = np.nan

    return dataframe


def similarity_of_anime_lists(features1, features2, metric="jaccard"):
    similarities = categorical_similarity(features1, features2, metric=metric)

    return similarities.mean(axis=1, skipna=True)


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
