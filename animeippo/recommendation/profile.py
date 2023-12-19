import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np

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
        # self.feature_correlations = self.get_feature_correlations()
        # self.last_liked = self.get_last_liked()

    def get_genre_correlations(self):
        if "genres" not in self.watchlist.columns:
            return None

        gdf = self.watchlist.explode("genres")

        return analysis.weight_categoricals_correlation(gdf, "genres").sort_values(ascending=False)

    def get_studio_correlations(self):
        if "studios" not in self.watchlist.columns:
            return None

        gdf = self.watchlist.explode("studios")

        return analysis.weight_categoricals_correlation(gdf, "studios").sort_values(ascending=False)

    def get_director_correlations(self):
        if "directors" not in self.watchlist.columns:
            return None

        gdf = self.watchlist.explode("directors")

        return analysis.weight_categoricals_correlation(gdf, "directors").sort_values(
            ascending=False
        )

    def get_feature_correlations(self, all_features):
        if "encoded" not in self.watchlist.columns:
            return None

        return analysis.weight_encoded_categoricals_correlation(
            self.watchlist, "encoded", all_features
        )

    def get_cluster_correlations(self):
        if "cluster" not in self.watchlist.columns:
            return None

        return analysis.weight_categoricals_correlation(self.watchlist, "cluster")


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

        all_features = user_profile.watchlist.explode("features")["features"].dropna().unique()

        self.encoder.fit(all_features)

        user_profile.watchlist["encoded"] = self.encoder.encode(user_profile.watchlist)

        encoded = np.stack(user_profile.watchlist["encoded"].values)
        user_profile.watchlist["cluster"] = self.clusterer.cluster_by_features(
            encoded, user_profile.watchlist.index
        )

        data = dataset.RecommendationModel(user_profile, None, all_features)
        data.nsfw_tags = self.get_nsfw_tags(data.watchlist)

        return data

    def get_nsfw_tags(self, df):
        if "nsfw_tags" in df.columns:
            return df["nsfw_tags"].explode().dropna().unique().tolist()

        return []

    def analyse(self, user):
        self.dataset = self.async_get_dataset(user)

        return self.get_categories(self.dataset)

    def get_categories(self, dataset):
        categories = []
        top_genre_items = []
        top_tag_items = []

        if "genres" in dataset.watchlist.columns:
            top_5_genres = dataset.user_profile.genre_correlations.iloc[0:5].index.tolist()

            for genre in top_5_genres:
                gdf = dataset.watchlist_explode_cached("genres")
                filtered = gdf[gdf["genres"] == genre].sort_values("score", ascending=False)

                if len(filtered) > 0:
                    top_genre_items.append(int(filtered.iloc[0].name))

            categories.append({"name": ", ".join(top_5_genres), "items": top_genre_items})

        if "tags" in dataset.watchlist.columns:
            tag_correlations = analysis.weight_categoricals_correlation(
                dataset.watchlist.explode("tags"), "tags"
            ).sort_values(ascending=False)

            top_5_tags = tag_correlations.iloc[0:5].index.tolist()

            for tag in top_5_tags:
                gdf = dataset.watchlist.drop(top_genre_items).drop(top_tag_items).explode("tags")
                filtered = gdf[gdf["tags"] == tag].sort_values("score", ascending=False)

                if len(filtered) > 0:
                    top_tag_items.append(int(filtered.iloc[0].name))

            categories.append({"name": ", ".join(top_5_tags), "items": top_tag_items})

        return categories

    def get_cluster_categories(self, dataset):
        target = dataset.watchlist

        gdf = dataset.watchlist_explode_cached("features")

        gdf = gdf[~gdf["features"].isin(dataset.nsfw_tags)]

        descriptions = pdutil.extract_features(gdf["features"], gdf["cluster"], 2)

        clustergroups = target.sort_values(["title"]).groupby("cluster")

        return [
            {"name": " ".join(descriptions.iloc[key].tolist()), "items": value.tolist()}
            for key, value in clustergroups.groups.items()
        ]
