import polars as pl

from animeippo.analysis import statistics

CLUSTERING_DEFAULTS = {
    "distance_metric": "cosine",
    "distance_threshold": 0.63,
    "linkage": "average",
    "min_cluster_size": 3,
    "franchise_reduction": True,
}


class ProfileAnalyser:
    """Clusters a user watchlist titles to clusters of similar anime."""

    def __init__(self, recommender):
        self.recommender = recommender

        if self.recommender:
            self.provider = recommender.provider

    async def analyse(self, user, year=None, season=None):
        # TODO: Should maybe really allow year=None, season=None
        # TODO: This function naming does not really now match the context
        dataset = await self.recommender.recommend_seasonal_anime(year, season, user)

        show_clusters = self.group_watchlist_by_cluster(dataset)

        dataset.recommendations = self.filter_items_on_watchlist(
            dataset.recommendations, dataset.watchlist
        )
        show_clusters = show_clusters.join(
            self.group_recommendations_by_cluster(dataset.recommendations, dataset.watchlist),
            on="cluster",
            how="left",
        )

        show_clusters = self.merge_clusters_with_duplicate_names(show_clusters)

        return dataset.user_profile, show_clusters, dataset.recommendations

    def group_recommendations_by_cluster(self, recommendations, watchlist):
        recommendations = self.assign_continuations_to_prequel_cluster(recommendations, watchlist)

        return (
            recommendations.sort("cluster_similarity", descending=True)
            .group_by("cluster")
            .agg(pl.col("id").alias("recommendations"))
        )

    def assign_continuations_to_prequel_cluster(self, recommendations, watchlist):
        """
        If a seasonal item is a sequel to a watchlist item, use the prequel's cluster.
        We probably could and should use franchise info here, since it exists partially,
        but it's such a slop mess right now that it needs refactoring to be useful here.
        """
        if "continuation_to" not in recommendations.columns or recommendations[
            "continuation_to"
        ].dtype == pl.List(pl.Null):
            return recommendations

        wl_clusters = watchlist.select("id", "cluster")

        return (
            recommendations.explode("continuation_to", empty_as_null=True)
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

    def filter_items_on_watchlist(self, seasonal, watchlist):
        return seasonal.join(watchlist.select("id"), on="id", how="anti")

    def get_profile_top_genre_and_tag_categories(self, profile):
        """
        Get top genre and tag highlights for the profile page.
        Not in use right now, consider removing.
        """
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

    def group_watchlist_by_cluster(self, dataset):
        clusters = dataset.clusters
        watchlist = dataset.watchlist

        return (
            (
                watchlist.sort("user_status", "title")
                .group_by("cluster")
                .agg("id")
                .join(
                    clusters.select("cluster", "name"),
                    on="cluster",
                    how="left",
                )
                .join(
                    # TODO: Why completed_count is calculated but not used?
                    clusters.select("cluster", "count", "mean_score", "completion_rate"),
                    on="cluster",
                    how="left",
                )
            )
            .select(
                pl.col("cluster"),
                pl.col("name"),
                pl.col("id").alias("items"),
                pl.col("count"),
                pl.col("mean_score"),
                pl.col("completion_rate"),
            )
            .sort(pl.col("items").list.len(), descending=True)
        )

    def merge_clusters_with_duplicate_names(self, clusters):
        return (
            clusters.group_by("name")
            .agg(
                pl.col("cluster").first(),
                pl.col("items").explode(),
                # TODO: Could just send these as plain but front requires a nested dict
                pl.struct(
                    pl.col("count").sum(),
                    # Need weighted values for merged clusters, otherwise differently sized clusters
                    # would get distorted stats.
                    (pl.col("mean_score") * pl.col("count")).sum() / pl.col("count").sum(),
                    (pl.col("completion_rate") * pl.col("count")).sum() / pl.col("count").sum(),
                ).alias("stats"),
                pl.col("recommendations").explode(),
            )
            .sort(pl.col("items").list.len(), descending=True)
        )
