import polars as pl
import numpy as np
import scipy.spatial.distance as scdistance


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


def categorical_similarity(features1, features2, metric="jaccard", columns=None):
    """Calculate similarity between two series of categorical features. Assumes a series
    that contains vector-encoded representation of features."""
    similarities = pl.DataFrame(
        similarity(
            # Polars seems to have a bug where to_numpy gets a cached value
            # instead of the actual conversion, thus np.array(x.to_list()),
            # not to_numpy().
            np.stack(features1.to_list()),
            np.stack(features2.to_list()),
            metric=metric,
        )
    )

    if columns is not None:
        similarities.columns = columns

    return similarities


def similarity_of_anime_lists(features1, features2, metric="jaccard"):
    similarities = categorical_similarity(features1, features2, metric=metric)

    return similarities.mean_horizontal(ignore_nulls=True)


def mean_score_per_categorical(dataframe, column):
    return dataframe.group_by(column, maintain_order=True).agg(pl.col("score").mean())


def weighted_mean_for_categorical_values(dataframe, column, weights, fillna=0.0):
    if weights is None or len(weights) == 0:
        return fillna

    return (
        dataframe.select(
            pl.col("id"),
            pl.col(column).replace(
                old=weights["name"],
                new=weights["weight"],
                default=fillna,
            ),
        )
        .group_by("id", maintain_order=True)
        .agg(pl.col(column).mean())[column]
    )


def weighted_sum_for_categorical_values(dataframe, column, weights, fillna=0.0):
    if weights is None or len(weights) == 0:
        return fillna

    return (
        dataframe.select(
            pl.col("id"),
            pl.col(column).replace(
                old=weights["name"],
                new=weights["weight"],
                default=fillna,
            ),
        )
        .group_by("id", maintain_order=True)
        .agg(pl.col(column).sum())[column]
    )


def weight_categoricals(dataframe, column):
    averages = mean_score_per_categorical(dataframe, column)

    return (
        dataframe[column]
        .value_counts()
        .join(averages, on=column, how="left")
        .select(
            pl.col(column).alias("name"), (pl.col("count").sqrt() * pl.col("score")).alias("weight")
        )
        .sort("name")
    )


def weight_encoded_categoricals_correlation(dataframe, column, features, against="score"):
    """
    Weights running-length encoded categorical features by their correlation with the score.
    Assumes that the encoding is done with features sorted alphabetically.
    """

    df_non_na = dataframe.filter(pl.col(against).is_not_null())

    values = np.stack(df_non_na[column])
    scores = np.array(df_non_na[against])

    correlations = np.corrcoef(np.hstack((values, scores.reshape(-1, 1))), rowvar=False)[:-1, -1]

    return pl.DataFrame({"name": sorted(features), "weight": correlations}).fill_nan(0.0)


def weight_categoricals_correlation(dataframe, column, against=None):
    dataframe = dataframe.filter(pl.col(column).is_not_null())
    against = against if against is not None else dataframe["score"]

    weights = (
        dataframe[column]
        .value_counts()
        .sort(pl.col(column).cast(pl.Utf8))
        .select(column, pl.col("count").sqrt().alias("weight"))  # Lessen the effect of outliers
    )

    correlations = (
        dataframe.select(column)
        .to_dummies()  # Convert to marker variables so that we can correlate with 1 and 0
        .with_columns(score=against)  # Score or other variable to correlate with
        .filter(pl.col("score").is_not_null())  # Remove nulls, not scored items are not interesting
        .select(pl.corr(pl.exclude("score"), pl.col("score")))  # Correlate with score
        .transpose()  # Transpose to get correlations as rows
        .select(pl.col("column_0").mul(weights["weight"]).alias("weight"))  # Multiply with weights
        .fill_nan(0.0)
        .with_columns(name=weights[column])  # Add categorical names
    )

    return correlations


def normalize_column(df_column):
    return (df_column - df_column.min()) / (df_column.max() - df_column.min())

    # return skpre.minmax_scale(df_column)


def get_mean_score(compare_df, default=0):
    mean_score = compare_df.select("score").mean().item()

    if mean_score == 0 or mean_score is None or np.isnan(mean_score):
        mean_score = default

    return mean_score


def idymax(dataframe):
    return pl.DataFrame(
        {
            "idymax": dataframe.select(
                pl.concat_list(pl.col("id").gather(pl.exclude("id").arg_max())),
            ).item(),
            "max": dataframe.select(pl.concat_list(pl.exclude("id").max())).item(),
        }
    )
