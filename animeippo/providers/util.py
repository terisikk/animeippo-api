import polars as pl


def transform_to_animeippo_format(original, feature_names, schema, mapping):
    df = pl.DataFrame(schema=schema)

    if len(original) == 0:
        return df

    if "id" in original.columns:
        original = original.unique(subset=["id"], keep="first")

    df = run_mappers(df, original, mapping)

    existing_feature_columns = set(feature_names).intersection(df.columns)

    if len(existing_feature_columns) > 0:
        df = df.with_columns(features=get_features(df, existing_feature_columns))

    if "temp_ranks" in df.columns and df["temp_ranks"].dtype != pl.Null:
        df = df.with_columns(ranks=get_ranks(df).to_series())

    return df


def run_mappers(dataframe, original, mapping):
    return pl.DataFrame().with_columns(
        **{key: mapper.map(original) for key, mapper in mapping.items() if key in dataframe.columns}
    )


def get_features(dataframe, columns):
    return dataframe.select(pl.concat_list(columns).list.sort().list.drop_nulls()).to_series()


def get_ranks(df):
    return (
        df.select("id", pl.col("temp_ranks"))
        .explode("temp_ranks")
        .unnest("temp_ranks")
        .pivot(index="id", values="rank", on="name")
        .join(
            df.explode("genres")
            .select("id", "genres", pl.lit(100).alias("rank"))
            .pivot(index="id", values="rank", on="genres"),
            on="id",
            how="left",
        )
        .fill_null(0)
        .select(pl.struct(pl.exclude("id", "fake", "null", "null_right")))
    )
