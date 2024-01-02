import pandas as pd
import numpy as np
import scipy.stats as scstats


def extract_features(features, columns, n_features=None):
    """
    Extracts the most correlated features from a categorical column.
    Used to find the most correlated features for a given cluster.

    Creating a crosstab in polars is much more inconvenient than in pandas
    currently, so we convert to pandas here until I find a better way.
    """
    fetures = features.to_pandas()
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


def calculate_residuals(contingency_table, expected):
    residuals = ((contingency_table - expected) * np.abs((contingency_table - expected))) / np.sqrt(
        expected
    )
    return residuals
