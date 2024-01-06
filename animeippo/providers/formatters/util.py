import polars as pl
from ..anilist_data import ALL_GENRES


def combine_dataframes(dataframes):
    return pl.concat(dataframes)


def transform_to_animeippo_format(original, feature_names, keys, mapping):
    df = pl.DataFrame(schema=keys)

    if len(original) == 0:
        return df

    if "id" in original.columns:
        original = original.unique(subset=["id"])

    df = run_mappers(df, original, mapping)

    existing_feature_columns = set(feature_names).intersection(df.columns)

    if len(existing_feature_columns) > 0:
        df = df.with_columns(features=get_features(df, existing_feature_columns))

    if "ranks" in df.columns:
        df = df.with_columns(ranks=df.select(["ranks", "features"]).map_rows(get_ranks).to_series())

    return df


def run_mappers(dataframe, original, mapping):
    return pl.DataFrame().with_columns(
        **{key: mapper.map(original) for key, mapper in mapping.items() if key in dataframe.columns}
    )


def get_features(dataframe, columns):
    return dataframe.select(pl.concat_list(columns).list.sort().list.drop_nulls()).to_series()


def get_ranks(row):
    rank_mapping = row[0]
    features = row[1]
    all_ranks = []

    if rank_mapping is None:
        all_ranks = [100] * len(features)

    for feature in features:
        all_ranks.append(rank_mapping.get(feature, 100))

    return (all_ranks,)


def get_score(score):
    return score if score != 0 else None


def get_season(dataframe):
    return dataframe.select(
        pl.concat_str(["seasonYear", "season"], separator="/").str.to_lowercase()
    ).to_series()
