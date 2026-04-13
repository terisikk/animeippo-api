import polars as pl


def filter_continuation(seasonal, watchlist_ids):
    """Filter sequels — only keep items that are continuations of watched anime,
    standalone titles, or titles already on the user's list."""
    if "continuation_to" not in seasonal.columns:
        return seasonal

    if seasonal["continuation_to"].dtype == pl.List(pl.Null):
        return seasonal

    mask = (
        (pl.col("continuation_to").list.set_intersection(watchlist_ids) != [])
        | (pl.col("continuation_to") == [])
        | (pl.col("user_status").is_not_null())
    )

    return seasonal.filter(mask)


def transform_to_animeippo_format(original, schema, mapping):
    if len(original) == 0:
        return pl.DataFrame(schema=schema)

    df = run_mappers(original, mapping, schema)

    if "feature_info" in df.columns:
        df = df.with_columns(
            features=df["feature_info"]
            .list.eval(pl.element().struct.field("name"))
            .cast(pl.List(pl.Categorical)),
            tags=df["feature_info"]
            .list.eval(
                pl.when(pl.element().struct.field("category") != "Genre").then(
                    pl.element().struct.field("name")
                )
            )
            .list.drop_nulls(),
        )
        df = df.with_columns(clustering_ranks=get_clustering_ranks(df).to_series())

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


def get_clustering_ranks(df):
    """Build weighted feature vector for clustering from feature_info."""
    exploded = (
        df.select("id", "feature_info")
        .explode("feature_info")
        .filter(pl.col("feature_info").is_not_null())
        .unnest("feature_info")
    )

    is_genre = pl.col("category") == "Genre"
    result = df.select("id")

    # Tags: weight by category
    tag_ranks = (
        exploded.filter(~is_genre)
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
    result = result.join(tag_ranks, on="id", how="left")

    # Genres: weight by inverse document frequency (IDF needs at least one genre)
    genre_data = exploded.filter(is_genre)
    if len(genre_data) > 0:
        genre_counts = genre_data.group_by("name").len()
        idf_raw = (pl.lit(len(df)) / pl.col("len")).log()
        max_idf = pl.max_horizontal(idf_raw.max(), pl.lit(1.0))

        genre_idf = genre_counts.with_columns(
            idf_weight=(idf_raw / max_idf * GENRE_MAX_WEIGHT).cast(pl.UInt8)
        )
        genre_ranks = (
            genre_data.select("id", "name")
            .join(genre_idf, on="name")
            .pivot(index="id", values="idf_weight", on="name", aggregate_function="first")
        )
        result = result.join(genre_ranks, on="id", how="left")

    return result.fill_null(0).select(pl.struct(pl.exclude("id")))
