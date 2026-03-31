import polars as pl

from animeippo.analysis import encoding, statistics
from animeippo.clustering import model
from animeippo.meta.meta import run_coroutine
from animeippo.profiling.model import UserProfile
from animeippo.providers.util import filter_continuation
from animeippo.recommendation.cluster_naming import get_cluster_stats, name_all_clusters


class ProfileAnalyser:
    """Clusters a user watchlist titles to clusters of similar anime."""

    def __init__(self, provider):
        self.provider = provider
        self.encoder = encoding.WeightedCategoricalEncoder()
        self.seasonal = None
        self.clusterer = model.AnimeClustering(
            distance_metric="cosine",
            distance_threshold=0.63,
            linkage="average",
            min_cluster_size=3,
            franchise_reduction=True,
        )

    async def databuilder(self, user, year=None, season=None):
        user_watchlist = await self.provider.get_user_anime_list(user)
        user_profile = UserProfile(user, user_watchlist)

        seasonal = None
        if year is not None:
            seasonal = await self.provider.get_seasonal_anime_list(year, season)

        all_features = user_profile.watchlist.explode("features")["features"].unique().drop_nulls()

        self.encoder.fit(all_features)
        user_profile.watchlist = user_profile.watchlist.with_columns(
            encoded=self.encoder.encode(user_profile.watchlist)
        )

        user_profile.watchlist = user_profile.watchlist.with_columns(
            cluster=self.clusterer.cluster_by_features(user_profile.watchlist)
        )

        return user_profile, seasonal

    def analyse(self, user, year=None, season=None):
        self.profile, seasonal = run_coroutine(self.databuilder(user, year, season))

        categories = self.get_cluster_categories(self.profile)

        if seasonal is not None:
            self.add_seasonal_recommendations(categories, seasonal)

        return categories

    def add_seasonal_recommendations(self, categories, seasonal):
        seasonal = self.filter_seasonal(seasonal)

        watchlist_ids = self.profile.watchlist["id"].to_list()
        seasonal = seasonal.filter(~pl.col("id").is_in(watchlist_ids))

        seasonal = seasonal.with_columns(encoded=self.encoder.encode(seasonal))
        predictions = self.clusterer.predict(seasonal["encoded"])
        seasonal = seasonal.with_columns(
            cluster=predictions["cluster"],
            cluster_similarity=predictions["similarity"],
        )

        if "continuation_to" in seasonal.columns:
            seasonal = self.assign_continuations_to_prequel_cluster(seasonal)

        self.seasonal = seasonal

        cluster_map = {}
        for cat in categories:
            for item_id in (
                self.profile.watchlist.filter(pl.col("id").is_in(cat["items"]))["cluster"]
                .unique()
                .to_list()
            ):
                cluster_map[item_id] = cat

        for cluster_id, group in seasonal.sort("cluster_similarity", descending=True).group_by(
            "cluster"
        ):
            cluster_map[cluster_id[0]]["recommendations"] = group["id"].to_list()

    def assign_continuations_to_prequel_cluster(self, seasonal):
        """If a seasonal item is a sequel to a watchlist item, use the prequel's cluster."""
        if seasonal["continuation_to"].dtype == pl.List(pl.Null):
            return seasonal

        wl_clusters = self.profile.watchlist.select("id", "cluster")

        return (
            seasonal.explode("continuation_to")
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

    def filter_seasonal(self, seasonal):
        """Filter seasonal list the same way as recommendations."""
        watchlist = self.profile.watchlist

        previously_watched = watchlist.filter(
            pl.col("user_status").is_in(["COMPLETED", "CURRENT", "PAUSED"])
        )["id"].to_list()

        seasonal = seasonal.join(watchlist.select(["id", "user_status"]), on="id", how="left")

        return filter_continuation(seasonal, previously_watched)

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
            ).sort("weight", descending=True)

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
                "name": cluster_names.get(str(key[0]), ""),
                "items": value["id"].to_list(),
                "stats": cluster_stats.filter(pl.col("cluster") == key[0])
                .select(["count", "mean_score", "completion_rate"])
                .to_dicts()[0],
            }
            for key, value in clustergroups
        ]
