import asyncio
from concurrent.futures import ThreadPoolExecutor

import polars as pl

from animeippo.recommendation import clustering, encoding, dataset, analysis, util as pdutil


class UserProfile:
    def __init__(self, user, watchlist):
        self.user = user
        self.watchlist = watchlist
        self.mangalist = None
        self.last_liked = None
        self.genre_correlations = None
        self.director_correlations = None
        self.studio_correlations = None

        if self.watchlist is not None and "score" in self.watchlist.columns:
            self.fit()

    def fit(self):
        self.genre_correlations = self.get_genre_correlations()

        self.director_correlations = self.get_director_correlations()
        self.studio_correlations = self.get_studio_correlations()
        self.last_liked = self.get_last_liked()

    def get_last_liked(self):
        if "user_complete_date" not in self.watchlist.columns:
            return None

        mask = (
            pl.col("score").ge(pl.col("score").mean()) & pl.col("user_complete_date").is_not_null()
        )

        return self.watchlist.filter(mask).sort("user_complete_date", descending=True).head(10)

    def get_genre_correlations(self):
        if "genres" not in self.watchlist.columns:
            return None

        gdf = self.watchlist.explode("genres")

        return analysis.weight_categoricals_correlation(gdf, "genres").sort(
            "weight", descending=True
        )

    def get_studio_correlations(self):
        if "studios" not in self.watchlist.columns:
            return None

        gdf = self.watchlist.explode("studios")

        return analysis.weight_categoricals_correlation(gdf, "studios").sort(
            "weight", descending=True
        )

    def get_director_correlations(self):
        if "directors" not in self.watchlist.columns:
            return None

        gdf = self.watchlist.explode("directors")

        return analysis.weight_categoricals_correlation(gdf, "directors").sort(
            "weight", descending=True
        )


class ProfileAnalyser:
    """Clusters a user watchlist titles to clusters of similar anime."""

    def __init__(self, provider):
        self.provider = provider
        self.encoder = encoding.WeightedCategoricalEncoder()
        self.clusterer = clustering.AnimeClustering(
            distance_metric="cosine", distance_threshold=0.65, linkage="average"
        )

    def async_get_dataset(self, user):
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

        data = dataset.RecommendationModel(user_profile, None, all_features)
        data.nsfw_tags = self.provider.get_nsfw_tags()

        return data

    def analyse(self, user):
        self.dataset = self.async_get_dataset(user)

        return self.get_categories(self.dataset)

    def get_categories(self, dataset):
        categories = []
        top_genre_items = []
        top_tag_items = []

        if (
            "genres" in dataset.watchlist.columns
            and dataset.user_profile.genre_correlations is not None
        ):
            top_5_genres = dataset.user_profile.genre_correlations[0:5]["name"].to_list()

            for genre in top_5_genres:
                gdf = dataset.watchlist_explode_cached("genres")
                filtered = gdf.filter(pl.col("genres") == genre).sort("score", descending=True)

                if len(filtered) > 0:
                    top_genre_items.append(filtered.item(0, "id"))

            categories.append({"name": ", ".join(top_5_genres), "items": top_genre_items})

        if "tags" in dataset.watchlist.columns:
            tag_correlations = analysis.weight_categoricals_correlation(
                dataset.watchlist.explode("tags"), "tags"
            ).sort("weight", descending=True)

            top_5_tags = tag_correlations[0:5]["name"].to_list()

            for tag in top_5_tags:
                gdf = dataset.watchlist.filter(
                    ~pl.col("id").is_in(top_genre_items.extend(top_tag_items))
                ).explode("tags")
                filtered = gdf.filter(pl.col("tags") == tag).sort("score", descending=True)

                if len(filtered) > 0:
                    top_tag_items.append(str(filtered.item(0, "id")))

            categories.append({"name": ", ".join(top_5_tags), "items": top_tag_items})

        return categories

    def get_cluster_categories(self, dataset):
        target = dataset.watchlist

        gdf = dataset.watchlist_explode_cached("features")

        gdf = gdf.filter(~pl.col("features").is_in(dataset.nsfw_tags))

        descriptions = pdutil.extract_features(gdf["features"], gdf["cluster"], 2)

        clustergroups = target.sort("title").group_by("cluster")

        return [
            {
                "name": " ".join(descriptions.iloc[key].tolist()),
                "items": value["id"].to_list(),
            }
            for key, value in clustergroups
        ]
