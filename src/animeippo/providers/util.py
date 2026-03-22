import math

import polars as pl


def transform_to_animeippo_format(original, feature_names, schema, mapping):
    if len(original) == 0:
        return pl.DataFrame(schema=schema)

    df = run_mappers(original, mapping, schema)

    existing_feature_columns = set(feature_names).intersection(schema.keys())

    if len(existing_feature_columns) > 0:
        df = df.with_columns(features=get_feature_selector(existing_feature_columns))

    if "temp_ranks" in schema.keys():
        df = df.with_columns(ranks=get_ranks(df).to_series())

    return df


# This here forgets the original schema and dtypes, so optimize this
def run_mappers(original, mapping, schema):
    return (
        pl.LazyFrame()
        .with_columns(
            **{
                key: (mapper.map(original).cast(schema.get(key)))
                for key, mapper in mapping.items()
                if key in schema
            }
        )
        .collect()
    )


def get_feature_selector(columns):
    return pl.concat_list(columns).cast(pl.List(pl.Categorical(ordering="lexical")))


# UnionFind is used instead of Polars joins for connected components because
# Polars has no graph primitives — a join-based label propagation approach
# was benchmarked at 2.2x slower due to repeated DataFrame creation overhead.
class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        while self.parent.get(x, x) != x:
            self.parent[x] = self.parent.get(self.parent[x], self.parent[x])
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_franchise_ids(ids, relation_lists):
    """Build franchise IDs from relation edges using union-find.

    Returns a List[Utf8] Series: ["franchise_<root_id>"] for anime in
    multi-member franchises, [] for singletons.
    """
    uf = UnionFind()
    id_list = ids.to_list()
    rel_list = relation_lists.to_list()

    for anime_id, relations in zip(id_list, rel_list, strict=True):
        if relations:
            for related_id in relations:
                uf.union(anime_id, related_id)

    root_counts = {}
    for anime_id in id_list:
        root = uf.find(anime_id)
        root_counts[root] = root_counts.get(root, 0) + 1

    return pl.Series(
        "franchise",
        [
            [f"franchise_{uf.find(aid)}"] if root_counts.get(uf.find(aid), 0) > 1 else []
            for aid in id_list
        ],
    )


GENRE_MAX_WEIGHT = 200

TAG_WEIGHTS = {
    "Theme": 1.5,
    "Setting": 1.5,
    "Cast": 0.5,
    "Cast-Main Cast": 1.5,
    "Demographic": 1.5,
    "Technical": 0.5,
    "Sexual Content": 0.5,
}


def get_ranks(df):
    # Process tags with category-based weights
    tag_ranks = (
        df.select("id", pl.col("temp_ranks"))
        .explode("temp_ranks")
        .unnest("temp_ranks")
        .with_columns(
            weighted_rank=pl.col("rank")
            * pl.col("category")
            .replace_strict(TAG_WEIGHTS, default=None)
            .fill_null(
                pl.col("category")
                .str.split("-")
                .list.first()
                .replace_strict(TAG_WEIGHTS, default=1.0)
            )
        )
        .pivot(index="id", values="weighted_rank", on="name", aggregate_function="first")
    )

    # Genre weight scaled by inverse frequency — rare genres in the user's watchlist
    # get higher weight than ubiquitous ones like Action
    genre_counts = df.select("genres").explode("genres").group_by("genres").len()

    max_idf = math.log(len(df) / genre_counts["len"].min()) or 1.0
    genre_idf = genre_counts.with_columns(
        idf_weight=((pl.lit(len(df)) / pl.col("len")).log() / max_idf * GENRE_MAX_WEIGHT).cast(
            pl.UInt8
        )
    )

    genre_ranks = (
        df.select("id", pl.col("genres"))
        .explode("genres")
        .join(genre_idf, on="genres")
        .pivot(index="id", values="idf_weight", on="genres", aggregate_function="first")
    )

    return (
        tag_ranks.join(genre_ranks, on="id", how="left")
        .fill_null(0)
        .select(pl.struct(pl.exclude("id")))
    )
