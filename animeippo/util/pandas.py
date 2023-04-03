import pandas as pd
import numpy as np
import sklearn.preprocessing as skpre
import scipy.stats as scstats


def normalize_column(df_column):
    shaped = df_column.to_numpy().reshape(-1, 1)
    return pd.DataFrame(skpre.MinMaxScaler().fit_transform(shaped))


def extract_features(features, columns, n_features):
    contingency_table = pd.crosstab(features, columns)

    _, _, _, expected = scstats.chi2_contingency(contingency_table)

    residuals = calculate_residuals(contingency_table, expected)

    descriptions = contingency_table.apply(
        lambda row: residuals.nlargest(n_features, row.name).index.values, axis=0
    ).T

    return descriptions


def calculate_residuals(contingency_table, expected):
    residuals = (contingency_table - expected) / np.sqrt(expected)
    return residuals


def one_hot_categorical(df_column, classes):
    mlb = skpre.MultiLabelBinarizer(classes=classes)
    mlb.fit(None)
    return np.array(mlb.transform(df_column), dtype=bool)
