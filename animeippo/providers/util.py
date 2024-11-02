import polars as pl


def transform_to_animeippo_format(original, feature_names, schema, mapping):
    df = pl.DataFrame(schema=schema)

    if len(original) == 0:
        return df

    if "id" in original.columns:
        original = original.unique(subset=["id"], keep="first")

    df = run_mappers(df, original, mapping, schema)

    existing_feature_columns = set(feature_names).intersection(df.columns)

    if len(existing_feature_columns) > 0:
        df = df.with_columns(features=get_features(df, existing_feature_columns))

    if "temp_ranks" in df.columns and df["temp_ranks"].dtype != pl.Null:
        df = df.with_columns(ranks=get_ranks(df).to_series())

    return df


# This here forgets the original schema and dtypes, so optimize this
def run_mappers(dataframe, original, mapping, schema):
    return pl.DataFrame().with_columns(
        **{
            key: (
                mapper.map(original).cast(schema.get(key))
                if key in schema
                else mapper.map(original)
            )
            for key, mapper in mapping.items()
            if key in dataframe.columns and key in schema
        }
    )


def get_features(dataframe, columns):
    return dataframe.select(pl.concat_list(columns).list.sort().list.drop_nulls()).to_series()


TAG_WEIGHTS = {
    "Theme": 1.5,
    "Setting": 1.5,
    "Cast": 0.5,
    "Demographic": 1.5,
    "Technical": 0.5,
    "Sexual Content": 0.2,
}


def get_ranks(df):
    return (
        df.select("id", pl.col("temp_ranks"))
        .explode("temp_ranks")
        .unnest("temp_ranks")
        .select(
            pl.col("*"),
            pl.col("category")
            .str.split("-")
            .alias("category_weight")
            .list.first()
            .replace_strict(TAG_WEIGHTS),
        )
        .select(pl.exclude("rank"), pl.col("rank") * pl.col("category_weight"))
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
