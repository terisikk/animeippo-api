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
    return 1 - distances  # This is incorrect for distances that are not 0-1


def categorical_similarity(features1, features2, index=None, metric="jaccard"):
    """Calculate similarity between two series of categorical features. Assumes a series
    that contains vector-encoded representation of features."""
    if index is None:
        index = features1.index

    similarities = pd.DataFrame(
        similarity(np.stack(features1.values), np.stack(features2.values), metric=metric),
        index=index,
        columns=features2.index,
    )

    return similarities.fillna(0.0)


def similarity_of_anime_lists(features1, features2, metric="jaccard"):
    similarities = categorical_similarity(features1, features2, metric=metric)

    return similarities.mean(axis=1, skipna=True)


def mean_score_per_categorical(dataframe, column):
    return dataframe.groupby(column)["score"].mean()


def weighted_mean_for_categorical_values(categoricals, weights, fillna=0.0):
    sum = 0.0
    lc = len(categoricals)

    for categorical in categoricals:
        sum += weights.get(categorical, fillna)

    return sum / lc if lc > 0 else 0.0


def weighted_sum_for_categorical_values(categoricals, weights, fillna=0.0):
    sum = 0.0
    for categorical in categoricals:
        sum += weights.get(categorical, fillna)

    return sum


def weight_categoricals(dataframe, column):
    averages = mean_score_per_categorical(dataframe, column)

    weights = np.sqrt(dataframe[column].value_counts())
    weights = weights * averages

    return weights


def weight_encoded_categoricals_correlation(dataframe, column, features, against=None):
    against = against.astype("float64") if against is not None else dataframe["score"]
    against = against[~pd.isna(against)]

    df_non_na = dataframe.loc[against.index]

    values = np.stack(df_non_na[column].values)
    scores = np.array(against.values)

    correlations = np.corrcoef(np.hstack((values, scores.reshape(-1, 1))), rowvar=False)[:-1, -1]

    return pd.Series(correlations, index=sorted(features)).fillna(0.0)


def weight_categoricals_correlation(dataframe, column, against=None):
    dummies = pd.get_dummies(dataframe[column], dtype=int)
    dummies["score"] = against if against is not None else dataframe["score"]

    dummies_non_na = dummies[~pd.isna(dummies["score"])]

    correlation_matrix = np.corrcoef(dummies_non_na, rowvar=False)[:-1, -1]

    weights = np.sqrt(dataframe[column].value_counts())

    correlations = correlation_matrix * weights.sort_index()

    return correlations.fillna(0.0)


def normalize_column(df_column):
    shaped = df_column.to_numpy().reshape(-1, 1)
    return pd.DataFrame(skpre.MinMaxScaler().fit_transform(shaped), index=df_column.index)


def get_mean_score(compare_df, default=0):
    mean_score = compare_df["score"].mean(skipna=True)

    if mean_score == 0 or pd.isna(mean_score):
        mean_score = default

    return mean_score
