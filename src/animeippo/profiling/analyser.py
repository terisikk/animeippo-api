import copy

import polars as pl

from animeippo.analysis import encoding, statistics
from animeippo.clustering import model
from animeippo.profiling.model import UserProfile
from animeippo.providers.util import filter_irrelevant_continuation
from animeippo.recommendation.cluster_naming import get_cluster_stats, name_all_clusters

CLUSTERING_DEFAULTS = {
    "distance_metric": "cosine",
    "distance_threshold": 0.63,
    "linkage": "average",
    "min_cluster_size": 3,
    "franchise_reduction": True,
}


class ProfileAnalyser:
    """Clusters a user watchlist titles to clusters of similar anime."""

    def __init__(self, provider, clustering_defaults=None):
        self.provider = provider
        self.encoder = encoding.WeightedCategoricalEncoder()
        self.clusterer = model.AnimeClustering(**(clustering_defaults or CLUSTERING_DEFAULTS))

    async def databuilder(self, user, year=None, season=None):
        # Fresh copies so concurrent requests don't share mutable fit state
        encoder = copy.copy(self.encoder)
        clusterer = copy.copy(self.clusterer)

        user_watchlist = await self.provider.get_user_anime_list(user)
        user_profile = UserProfile(user, user_watchlist)

        seasonal = None
        if year is not None:
            seasonal = await self.provider.get_seasonal_anime_list(year, season)

        all_features = user_profile.watchlist.explode("features")["features"].unique().drop_nulls()

        encoder.fit(all_features)
        user_profile.watchlist = user_profile.watchlist.with_columns(
            encoded=encoder.encode(user_profile.watchlist)
        )

        user_profile.watchlist = user_profile.watchlist.with_columns(
            cluster=clusterer.cluster_by_features(user_profile.watchlist)
        )

        return user_profile, seasonal, clusterer

    async def analyse(self, user, year=None, season=None):
        profile, seasonal, clusterer = await self.databuilder(user, year, season)

        categories = self.get_cluster_categories(profile.watchlist)

        if seasonal is not None:
            seasonal = self.add_seasonal_recommendations(profile, seasonal, clusterer)
            categories = self.populate_category_recommendations(seasonal, categories)

        return profile, categories, seasonal

    def add_seasonal_recommendations(self, profile, seasonal, clusterer):
        seasonal = self.filter_seasonal(profile, seasonal)
        seasonal = self.assign_clustering(profile, seasonal, clusterer)

        return seasonal

    def populate_category_recommendations(self, seasonal, categories):
        return categories.join(
            seasonal.sort("cluster_similarity", descending=True)
            .group_by("cluster")
            .agg(pl.col("id").alias("recommendations")),
            on="cluster",
            how="left",
        )

    def filter_seasonal(self, profile, seasonal):
        """Filter seasonal list the same way as recommendations."""
        watchlist = profile.watchlist

        seasonal = self.enrich_seasonal_with_user_status(seasonal, watchlist)
        seasonal = filter_irrelevant_continuation(seasonal, watchlist)
        seasonal = self.filter_items_on_watchlist(seasonal, watchlist)

        return seasonal

    def assign_clustering(self, profile, seasonal, clusterer):
        encoder = self.create_encoder(profile)
        seasonal = seasonal.with_columns(encoded=encoder.encode(seasonal))

        predictions = clusterer.predict(seasonal["encoded"])
        seasonal = seasonal.with_columns(
            cluster=predictions["cluster"],
            cluster_similarity=predictions["similarity"],
        )

        seasonal = self.assign_continuations_to_prequel_cluster(profile, seasonal)
        return seasonal

    def create_encoder(self, profile):
        encoder = copy.copy(self.encoder)
        encoder.fit(profile.watchlist.explode("features")["features"].unique().drop_nulls())
        return encoder

    def assign_continuations_to_prequel_cluster(self, profile, seasonal):
        """If a seasonal item is a sequel to a watchlist item, use the prequel's cluster."""
        if "continuation_to" not in seasonal.columns:
            return seasonal

        if seasonal["continuation_to"].dtype == pl.List(pl.Null):
            return seasonal

        wl_clusters = profile.watchlist.select("id", "cluster")

        return (
            seasonal.explode("continuation_to", empty_as_null=True)
            .join(
                wl_clusters.rename({"id": "continuation_to", "cluster": "prequel_cluster"}),
                on="continuation_to",
                how="left",
            )
            .group_by("id", maintain_order=True)
            .agg(
                pl.exclude("continuation_to", "prequel_cluster").first(),
                pl.col("prequel_cluster").drop_nulls().first(),
            )
            .with_columns(
                cluster=pl.when(pl.col("prequel_cluster").is_not_null())
                .then(pl.col("prequel_cluster"))
                .otherwise(pl.col("cluster"))
            )
            .drop("prequel_cluster")
        )

    def enrich_seasonal_with_user_status(self, seasonal, watchlist):
        return seasonal.join(watchlist.select(["id", "user_status"]), on="id", how="left")

    def filter_items_on_watchlist(self, seasonal, watchlist):
        return seasonal.join(watchlist.select("id"), on="id", how="anti")

    def get_profile_top_genre_and_tag_categories(self, profile):
        """Get top genre and tag highlights for the profile page."""
        categories = []
        top_genre_items = []
        top_tag_items = []

        if (
            "genres" in profile.watchlist.columns
            and profile.user_profile.genre_correlations is not None
        ):
            top_5_genres = profile.user_profile.genre_correlations[0:5]["name"].to_list()

            for genre in top_5_genres:
                gdf = profile.watchlist_explode_cached("genres")
                filtered = gdf.filter(pl.col("genres") == genre).sort("score", descending=True)

                if len(filtered) > 0:
                    top_genre_items.append(filtered.item(0, "id"))

            categories.append({"name": ", ".join(top_5_genres), "items": top_genre_items})

        if "tags" in profile.watchlist.columns:
            tag_correlations = statistics.weight_categoricals_correlation(
                profile.watchlist.explode("tags"), "tags"
            ).sort(["weight", "name"], descending=True)

            top_5_tags = tag_correlations[0:5]["name"].to_list()

            for tag in top_5_tags:
                gdf = profile.watchlist.filter(
                    ~pl.col("id").is_in(pl.Series(top_genre_items.extend(top_tag_items)).implode())
                ).explode("tags")
                filtered = gdf.filter(pl.col("tags") == tag).sort("score", descending=True)

                if len(filtered) > 0:
                    top_tag_items.append(str(filtered.item(0, "id")))

            categories.append({"name": ", ".join(top_5_tags), "items": top_tag_items})

        return categories

    def get_cluster_categories(self, watchlist):
        cluster_names = name_all_clusters(
            watchlist,
            tag_lookup=self.provider.get_tag_lookup(),
            genres=self.provider.get_genres(),
            nsfw_tags=self.provider.get_nsfw_tags(),
        )

        # TODO: Kinda stupid to convert back to dataframe here but
        # it's a temporary measure until we change cluster namer to
        # work on dataframes too, which it probably should
        cluster_names_df = pl.DataFrame(
            {"cluster": cluster_names.keys(), "name": cluster_names.values()}
        )

        cluster_stats = get_cluster_stats(watchlist)

        return (
            (
                watchlist.sort("user_status", "title")
                .group_by("cluster")
                .agg("id")
                .join(
                    cluster_names_df.select(pl.col("cluster").cast(pl.Int64), "name"),
                    on="cluster",
                    how="left",
                )
                .join(
                    # TODO: Why completed_count is calculated but not used?
                    cluster_stats.select("cluster", "count", "mean_score", "completion_rate"),
                    on="cluster",
                    how="left",
                )
            )
            .select(
                pl.col("cluster"),
                pl.col("name"),
                pl.col("id").alias("items"),
                # TODO: Could just send these as plain but front requires a nested dict
                pl.struct(pl.col("count"), pl.col("mean_score"), pl.col("completion_rate")).alias(
                    "stats"
                ),
            )
            .sort(pl.col("items").list.len(), descending=True)
        )
