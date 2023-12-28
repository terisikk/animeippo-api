import functools
import pandas as pd
import polars as pl
import numpy as np


def combine_dataframes(dataframes):
    return pd.concat(dataframes)


def transform_to_animeippo_format(original, feature_names, keys, mapping):
    df = pl.DataFrame(schema=keys)

    if len(original) == 0:
        return df

    df = run_mappers(df, original, mapping)

    df = df.with_columns(features=df.select(feature_names).map_rows(get_features).to_series())

    if "id" in df.columns:
        df = df.unique(subset=["id"])

    return df


def run_mappers(dataframe, original, mapping):
    return pl.DataFrame(
        {key: mapper.map(original) for key, mapper in mapping.items() if key in dataframe.columns}
    )


def default_if_error(default):
    def decorator_function(func):
        @functools.wraps(func)
        def wrapper(*args):
            try:
                return func(*args)
            except (TypeError, ValueError, AttributeError, KeyError) as error:
                print(error)
                return default

        return wrapper

    return decorator_function


def get_features(row):
    all_features = []

    for value in row:
        if value is None:
            continue

        if isinstance(value, list) or isinstance(value, np.ndarray):
            all_features.extend([v for v in value])
        else:
            all_features.append(value)

    return (all_features,)


def get_score(score):
    # np.nan is a float, pd.NA is not, causes problems
    return score if score != 0 else np.nan


def get_season(year, season):
    if year == 0 or pd.isna(year):
        year = "?"
    else:
        year = str(int(year))

    if pd.isna(season):
        season = "?"

    return (f"{year}/{str(season).lower()}",)
