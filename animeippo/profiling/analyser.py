import asyncio
from concurrent.futures import ThreadPoolExecutor

import polars as pl

from animeippo.analysis import encoding, statistics
from animeippo.clustering import model
from animeippo.profiling import characteristics
from animeippo.profiling.model import UserProfile


class ProfileAnalyser:
    """Clusters a user watchlist titles to clusters of similar anime."""

    def __init__(self, provider):
        self.provider = provider
        self.encoder = encoding.WeightedCategoricalEncoder()
        self.clusterer = model.AnimeClustering(
            distance_metric="cosine", distance_threshold=0.78, linkage="complete"
        )

    def async_get_profile(self, user):
        # If we run from jupyter, loop is already running and we need
        # to act differently. If the loop is not running,
        # we break into "normal path" with RuntimeError
        try:
            asyncio.get_running_loop()

            with ThreadPoolExecutor(1) as pool:
                return pool.submit(lambda: asyncio.run(self.databuilder(user))).result()
        except RuntimeError:
            return asyncio.run(self.databuilder(user))

    async def databuilder(self, user):
        user_watchlist = await self.provider.get_user_anime_list(user)
        user_profile = UserProfile(user, user_watchlist)

        all_features = user_profile.watchlist.explode("features")["features"].unique().drop_nulls()

        self.encoder.fit(all_features)
        user_profile.watchlist = user_profile.watchlist.with_columns(
            encoded=self.encoder.encode(user_profile.watchlist)
        )

        user_profile.watchlist = user_profile.watchlist.with_columns(
            cluster=self.clusterer.cluster_by_features(user_profile.watchlist)
        )

        return user_profile

    def analyse(self, user):
        self.profile = self.async_get_profile(user)

        self.profile.characteristics = characteristics.Characteristics(
            self.profile.watchlist, self.provider.get_genres()
        )

        return self.get_cluster_categories(self.profile)

    def get_categories(self, profile):
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
                    ~pl.col("id").is_in(top_genre_items.extend(top_tag_items))
                ).explode("tags")
                filtered = gdf.filter(pl.col("tags") == tag).sort("score", descending=True)

                if len(filtered) > 0:
                    top_tag_items.append(str(filtered.item(0, "id")))

            categories.append({"name": ", ".join(top_5_tags), "items": top_tag_items})

        return categories

    def get_cluster_categories(self, profile):
        target = profile.watchlist

        gdf = target.explode("features")

        gdf = gdf.filter(~pl.col("features").is_in(self.provider.get_nsfw_tags()))

        descriptions = statistics.get_descriptive_features(gdf, "features", "cluster", 2).select(
            pl.col("cluster"), pl.concat_list(pl.exclude("cluster")).alias("description")
        )

        clustergroups = target.sort("title").group_by(["cluster"])

        return [
            {
                "name": " ".join(
                    descriptions.filter(pl.col("cluster") == str(key[0]))["description"].item()
                ),
                "items": value["id"].to_list(),
            }
            for key, value in clustergroups
        ]
