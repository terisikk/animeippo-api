import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from animeippo.recommendation import clustering, encoding, dataset, util as pdutil


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

        data = dataset.UserDataSet(user_watchlist, None, None)

        data.all_features = data.watchlist_explode_cached("features")["features"].dropna().unique()

        self.encoder.fit(data.all_features)

        data.watchlist["encoded"] = self.encoder.encode(data.watchlist)

        encoded = np.stack(data.watchlist["encoded"].values)
        data.watchlist["cluster"] = self.clusterer.cluster_by_features(
            encoded, data.watchlist.index
        )

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
        target = dataset.watchlist

        gdf = dataset.watchlist_explode_cached("features")

        gdf = gdf[~gdf["features"].isin(dataset.nsfw_tags)]

        descriptions = pdutil.extract_features(gdf["features"], gdf["cluster"], 2)

        clustergroups = target.sort_values(["title"]).groupby("cluster")

        return [
            {"name": " ".join(descriptions.iloc[key].tolist()), "items": value.tolist()}
            for key, value in clustergroups.groups.items()
        ]
