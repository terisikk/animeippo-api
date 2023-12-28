import pandas as pd
import polars as pl
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


def categorical_similarity(features1, features2, metric="jaccard"):
    """Calculate similarity between two series of categorical features. Assumes a series
    that contains vector-encoded representation of features."""
    similarities = pl.DataFrame(similarity(np.stack(features1), np.stack(features2), metric=metric))

    return similarities.fill_nan(0.0)


def similarity_of_anime_lists(features1, features2, metric="jaccard"):
    similarities = categorical_similarity(features1, features2, metric=metric)

    return similarities.mean_horizontal(ignore_nulls=True)


def mean_score_per_categorical(dataframe, column):
    return dataframe.groupby(column).agg(pl.col("score").mean())


def weighted_mean_for_categorical_values(categoricals, weights, fillna=0.0):
    if len(categoricals) == 0 or categoricals is None:
        return fillna

    df = pl.DataFrame(categoricals)
    df.columns = [weights.columns[0]]

    return (
        df.join(weights, on=weights.columns[0], how="left")
        .select("weight")
        .fill_null(fillna)
        .mean()
        .item()
    )


def weighted_sum_for_categorical_values(categoricals, weights, fillna=0.0):
    df = pl.DataFrame(categoricals)
    df.columns = [weights.columns[0]]

    return (
        df.join(weights, on=weights.columns[0], how="left")
        .select("weight")
        .fill_null(fillna)
        .sum()
        .item()
    )


def weight_categoricals(dataframe, column):
    averages = mean_score_per_categorical(dataframe, column)

    counts = dataframe[column].value_counts()
    weights = np.sqrt(counts["count"])
    weights = weights * averages["score"]

    return pl.DataFrame({column: counts[column], "weight": weights})


def weight_encoded_categoricals_correlation(dataframe, column, features, against=None):
    if against is not None:
        dataframe = pl.concat([dataframe, pl.DataFrame(against.alias("against"))], how="horizontal")
        df_non_na = dataframe.filter(dataframe["against"].is_not_null())
    else:
        df_non_na = dataframe.filter(dataframe["score"].is_not_null())
        against = df_non_na["score"]

    values = np.stack(df_non_na[column])
    scores = np.array(against)

    correlations = np.corrcoef(np.hstack((values, scores.reshape(-1, 1))), rowvar=False)[:-1, -1]

    return pl.DataFrame({"feature": features, "weight": correlations}).fill_nan(0.0)


def weight_categoricals_correlation(dataframe, column, against=None):
    against = against if against is not None else dataframe["score"]

    dummies = dataframe[column].to_dummies()
    dummies = dummies.with_columns(score=against)

    dummies_non_na = dummies.filter(dummies["score"].is_not_null())

    correlation_matrix = np.corrcoef(dummies_non_na, rowvar=False)[:-1, -1]

    counts = dataframe[column].value_counts()
    weights = np.sqrt(counts["count"])

    correlations = np.nan_to_num(correlation_matrix * weights, nan=0.0)

    return pl.DataFrame({column: counts[column], "weight": correlations})


def normalize_column(df_column):
    return skpre.minmax_scale(df_column)


def get_mean_score(compare_df, default=0):
    mean_score = compare_df.select("score").mean().item()

    if mean_score == 0 or mean_score is None:
        mean_score = default

    return mean_score
