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

        categories = self.get_cluster_categories(profile)

        if seasonal is not None:
            seasonal = self.add_seasonal_recommendations(profile, seasonal, clusterer)
            self.populate_category_recommendations(seasonal, categories)

        return profile, categories, seasonal

    def add_seasonal_recommendations(self, profile, seasonal, clusterer):
        seasonal = self.filter_seasonal(profile, seasonal)
        seasonal = self.assign_clustering(profile, seasonal, clusterer)

        return seasonal

    def populate_category_recommendations(self, seasonal, categories):
        # FIXME: This is still quite side-effecty§

        cluster_map = {cat["cluster"]: cat for cat in categories}

        for cluster_id, group in seasonal.sort("cluster_similarity", descending=True).group_by(
            "cluster"
        ):
            # FIXME: What, why [0]
            cluster_map[cluster_id[0]]["recommendations"] = group["id"].to_list()

    def filter_seasonal(self, profile, seasonal):
        """Filter seasonal list the same way as recommendations."""
        watchlist = profile.watchlist

        previously_watched = watchlist.filter(
            pl.col("user_status").is_in(["COMPLETED", "CURRENT", "PAUSED"])
        )["id"].to_list()

        seasonal = self.enrich_seasonal_with_user_status(seasonal, watchlist)
        seasonal = filter_irrelevant_continuation(seasonal, previously_watched)
        seasonal = self.filter_items_on_watchlist(watchlist, seasonal)

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

    def filter_items_on_watchlist(self, watchlist, seasonal):
        watchlist_ids = watchlist["id"].to_list()
        seasonal = seasonal.filter(~pl.col("id").is_in(watchlist_ids))
        return seasonal

    def get_categories(self, profile):
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

    def get_cluster_categories(self, profile):
        target = profile.watchlist

        cluster_names = name_all_clusters(
            target,
            tag_lookup=self.provider.get_tag_lookup(),
            genres=self.provider.get_genres(),
            nsfw_tags=self.provider.get_nsfw_tags(),
        )

        cluster_stats = get_cluster_stats(target)

        clustergroups = target.sort("user_status", "title").group_by(["cluster"])

        return [
            {
                "cluster": key[0],
                "name": cluster_names.get(str(key[0]), ""),
                "items": value["id"].to_list(),
                "stats": cluster_stats.filter(pl.col("cluster") == key[0])
                .select(["count", "mean_score", "completion_rate"])
                .to_dicts()[0],
            }
            for key, value in clustergroups
        ]
