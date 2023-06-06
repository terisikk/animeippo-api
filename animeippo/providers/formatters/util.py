import functools
import pandas as pd
import numpy as np


def format_with_formatters(df, formatters):
    for key, formatter in formatters.items():
        if key in df.columns:
            df[key] = df[key].apply(formatter)

    return df


def get_column_name_mappers(columns):
    return {key: key.split(".")[-1] for key in columns if "." in key}


def default_if_error(default):
    def decorator_function(func):
        @functools.wraps(func)
        def wrapper(field):
            try:
                return func(field)
            except (TypeError, ValueError, AttributeError, KeyError) as error:
                print(error)
                return default

        return wrapper

    return decorator_function


def get_features(row, feature_names):
    all_features = []

    if feature_names is not None:
        for feature in feature_names:
            value = row.get(feature, None)

            if isinstance(value, list) or isinstance(value, np.ndarray):
                all_features.extend([v for v in value if not pd.isnull(v)])
            elif value is None or pd.isnull(value):
                continue
            else:
                all_features.append(value)

    return all_features
