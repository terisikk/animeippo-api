import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfTransformer


def weighted_mean_for_categorical_values(dataframe, column, weights, fillna=0.0):
    if weights is None or len(weights) == 0:
        return fillna

    return (
        dataframe.select(
            pl.col("id"),
            pl.col(column).replace_strict(
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
            pl.col(column).replace_strict(
                old=weights["name"],
                new=weights["weight"],
                default=fillna,
            ),
        )
        .group_by("id", maintain_order=True)
        .agg(pl.col(column).sum())[column]
    )


def weight_encoded_categoricals_correlation(dataframe, column, against="score", header_name="name"):
    return (
        dataframe.filter(pl.col(against).is_not_null())
        .select(
            pl.col(column),
            pl.col(against).alias("score"),
        )
        # One would think that .struct.unnest() would be faster, but it is not
        .unnest(column)
        .select(pl.corr(pl.exclude("score"), pl.col("score"), method="spearman"))
        .transpose(include_header=True, header_name=header_name, column_names=["weight"])
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
        .select(
            pl.corr(pl.exclude("score"), pl.col("score"), method="spearman")
        )  # Correlate with score
        .transpose(column_names=["weight"])  # Transpose to get correlations as rows
        .select(pl.col("weight").mul(weights["weight"]))  # Multiply with weights
        .fill_nan(0.0)
        .with_columns(name=weights[column])
    )


def rank_series(series):
    return (series.rank(method="average") - 1) / max(len(series) - 1, 1)  # Normalize to 0-1


def mean_score_default(compare_df, default=0):
    mean_score = compare_df.select("score").mean().item()

    if mean_score == 0 or mean_score is None or np.isnan(mean_score):
        mean_score = default

    return mean_score


def bounded_rating_modifier(score_col, default_score=5.0):
    """Rating as a 0.5-1.0 modifier so it influences but doesn't dominate similarity."""
    return 0.5 + 0.5 * score_col.fill_null(default_score).cast(pl.Float64) / 10.0


def weighted_top_k(values, weights):
    """Weighted average of the top-k values using pre-defined decay weights."""
    active_weights = weights[: len(values)]
    return sum(v * w for v, w in zip(values, active_weights, strict=False)) / sum(active_weights)


def calculate_residuals(contingency_table, expected):
    return ((contingency_table - expected) * np.abs(contingency_table - expected)) / np.sqrt(
        expected
    )


def get_descriptive_features(  # noqa: PLR0913
    dataframe,
    feature_column,
    cluster_column,
    n_features=None,
    min_count=2,
    min_prevalence=0.6,
    boost_features=None,
    boost_factor=1.5,
):
    """
    Extracts the most distinctive features using TF-IDF scoring.

    Uses sklearn's TfidfTransformer to score features by their distinctiveness to each cluster:
    - TF (term frequency): how common the feature is within the cluster
    - IDF (inverse document frequency): how rare the feature is across all clusters

    Features must appear in at least min_prevalence (60%) of a cluster's items
    to be eligible for naming that cluster.
    """

    # Build term-document matrix: features by clusters
    contingency_table = dataframe.pivot(
        on=cluster_column, index=feature_column, values=cluster_column, aggregate_function="len"
    ).fill_null(0)

    cluster_columns = contingency_table.columns[1:]

    # Zero out features that appear in fewer than min_prevalence of a cluster's items.
    # Cluster size = max feature count in that column (the most common feature appears
    # in every item, so its count equals the cluster size).
    if min_prevalence > 0:
        contingency_table = contingency_table.with_columns(
            pl.when(pl.col(col) >= pl.col(col).max() * min_prevalence)
            .then(pl.col(col))
            .otherwise(0)
            .alias(col)
            for col in cluster_columns
        )

    # Filter features that don't meet min_count in any cluster
    filtered = contingency_table.filter(
        pl.max_horizontal(pl.col(col) for col in cluster_columns) >= min_count
    )

    if len(filtered) > 0:
        contingency_table = filtered

    count_matrix = contingency_table.select(pl.exclude(feature_column)).to_numpy()

    # Apply TF-IDF transformation
    tfidf = TfidfTransformer()
    tfidf_matrix = tfidf.fit_transform(count_matrix.T).T  # Transpose: clusters as documents

    # Convert back to Polars with TF-IDF scores (convert sparse matrix to dense)
    tfidf_df = pl.concat(
        [
            contingency_table.select(pl.col(feature_column)),
            pl.DataFrame(tfidf_matrix.toarray(), schema=cluster_columns),
        ],
        how="horizontal",
    )

    # Boost preferred features (e.g. genres) so they win over tags when scores are close
    if boost_features:
        boost_mask = pl.col(feature_column).is_in(list(boost_features))
        tfidf_df = tfidf_df.with_columns(
            pl.when(boost_mask).then(pl.col(col) * boost_factor).otherwise(pl.col(col)).alias(col)
            for col in cluster_columns
        )

    if not n_features:
        n_features = len(tfidf_df)

    # Unpivot to long format, keep only non-zero scores, pick top N per cluster
    return (
        tfidf_df.unpivot(index=feature_column, variable_name="cluster", value_name="tfidf")
        .filter(pl.col("tfidf") > 0)
        .sort("tfidf", descending=True)
        .group_by("cluster", maintain_order=True)
        .head(n_features)
        .group_by("cluster", maintain_order=True)
        .agg(pl.col(feature_column).alias("description"))
    )


def catalogue_frequency(df: pl.DataFrame, list_col: str, value_col: str = "value") -> pl.DataFrame:
    """
    df[list_col] is a list column (e.g. 'genres' or 'features').
    Returns: value, pc (catalogue share in current season, in [0,1])
    """
    total = df.height
    return (
        df.select(list_col)
        .explode(list_col)
        .group_by(list_col)
        .len()
        .rename({list_col: value_col, "len": "cnt"})
        .with_columns((pl.col("cnt") / pl.lit(total)).alias("pc"))
        .select([value_col, "pc"])
    )
