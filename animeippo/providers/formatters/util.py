import functools
import polars as pl
import numpy as np


def combine_dataframes(dataframes):
    return pl.concat(dataframes)


def transform_to_animeippo_format(original, feature_names, keys, mapping):
    df = pl.DataFrame(schema=keys)

    if len(original) == 0:
        return df

    df = run_mappers(df, original, mapping)

    existing_feature_columns = set(feature_names).intersection(df.columns)

    if len(existing_feature_columns) > 0:
        df = df.with_columns(
            features=df.select(existing_feature_columns).map_rows(get_features).to_series()
        )

    if "ranks" in df.columns:
        df = df.with_columns(ranks=df.select(["ranks", "features"]).map_rows(get_ranks).to_series())

    if "id" in df.columns:
        df = df.unique(subset=["id"])

    return df


def run_mappers(dataframe, original, mapping):
    return pl.DataFrame().with_columns(
        **{key: mapper.map(original) for key, mapper in mapping.items() if key in dataframe.columns}
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

    return (sorted(all_features),)


def get_ranks(row):
    rank_mapping = row[0]
    features = row[1]
    all_ranks = []

    for feature in features:
        all_ranks.append(rank_mapping.get(feature, 100) if rank_mapping is not None else 1)

    return (all_ranks,)


def get_score(score):
    return score if score != 0 else None


def get_season(year, season):
    if year == 0 or year is None:
        year = "?"
    else:
        year = str(int(year))

    if season is None:
        season = "?"

    return (f"{year}/{str(season).lower()}",)
