import numpy as np
import pandas as pd
import polars as pl
import scipy.stats as scstats


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
    return (
        dataframe.filter(pl.col(against).is_not_null())
        .select(
            pl.col(column).list.to_struct(fields=sorted(features)),
            pl.col(against).alias("score"),
        )
        .unnest(column)
        .select(pl.corr(pl.exclude("score"), pl.col("score")))
        .transpose(include_header=True, header_name="name", column_names=["weight"])
        .fill_nan(0.0)
    )


def weight_categoricals_correlation(dataframe, column):
    dataframe = dataframe.filter(pl.col(column).is_not_null() & pl.col("score").is_not_null())

    if len(dataframe) == 0:
        return pl.DataFrame({"name": [], "weight": []})

    weights = (
        dataframe[column]
        .value_counts()
        .sort(pl.col(column).cast(pl.Utf8))
        .select(column, pl.col("count").sqrt().alias("weight"))  # Lessen the effect of outliers
    )

    return (
        dataframe.select(column, "score")
        .to_dummies(column)  # Convert to marker variables so that we can correlate with 1 and 0
        .select(pl.corr(pl.exclude("score"), pl.col("score")))  # Correlate with score
        .transpose(column_names=["weight"])  # Transpose to get correlations as rows
        .select(pl.col("weight").mul(weights["weight"]))  # Multiply with weights
        .fill_nan(0.0)
        .with_columns(name=weights[column])
    )


def normalize_series(series):
    return (series - series.min()) / (series.max() - series.min())


def mean_score_per_categorical(dataframe, column):
    return dataframe.group_by(column).agg(pl.col("score").mean())


def mean_score_default(compare_df, default=0):
    mean_score = compare_df.select("score").mean().item()

    if mean_score == 0 or mean_score is None or np.isnan(mean_score):
        mean_score = default

    return mean_score


def idymax(dataframe):
    return dataframe.select(
        pl.concat_list(pl.col("id").gather(pl.exclude("id").arg_max())).alias("idymax"),
        pl.concat_list(pl.exclude("id").max()).alias("max"),
    ).explode(["idymax", "max"])


def calculate_residuals(contingency_table, expected):
    residuals = ((contingency_table - expected) * np.abs((contingency_table - expected))) / np.sqrt(
        expected
    )
    return residuals


def extract_features(features, columns, n_features=None):
    """
    Extracts the most correlated features from a categorical column.
    Used to find the most correlated features for a given cluster.

    Creating a crosstab in polars is much more inconvenient than in pandas
    currently, so we convert to pandas here until I find a better way.
    """
    features = features.to_pandas()
    columns = columns.to_pandas()

    contingency_table = pd.crosstab(features, columns)

    _, _, _, expected = scstats.chi2_contingency(contingency_table)

    residuals = calculate_residuals(contingency_table, expected)

    if not n_features:
        n_features = len(residuals)

    descriptions = contingency_table.apply(
        lambda row: residuals.nlargest(n_features, row.name).index.values, axis=0
    ).T

    return descriptions
