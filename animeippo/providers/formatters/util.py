import functools
import pandas as pd
import numpy as np


def transform_to_animeippo_format(original, feature_names, keys, mapping):
    df = pd.DataFrame(columns=keys)

    if len(original) == 0:
        return df

    df = run_mappers(df, original, mapping)

    df["features"] = df.apply(get_features, args=(feature_names,), axis=1)

    if "id" in df.columns:
        df = df.drop_duplicates(subset="id")
        df = df.set_index("id")

    return df.infer_objects()


def run_mappers(dataframe, original, mapping):
    for key, mapper in mapping.items():
        if key in dataframe.columns:
            dataframe[key] = mapper.map(original)

    return dataframe


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

    return f"{year}/{str(season).lower()}"
