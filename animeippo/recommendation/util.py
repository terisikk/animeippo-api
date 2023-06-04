import pandas as pd
import numpy as np
import scipy.stats as scstats


def extract_features(features, columns, n_features=None):
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
    residuals = (contingency_table - expected) / np.sqrt(expected)
    return residuals
