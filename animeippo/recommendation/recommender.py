import asyncio
from concurrent.futures import ThreadPoolExecutor

import polars as pl


class AnimeRecommender:
    """Recommends new anime to a user if provided,
    or returns a filtered list of seasonal anime
    not tailored to a specific user."""

    def __init__(
        self,
        *,
        provider=None,
        engine=None,
        recommendation_model_cls=None,
        profile_model_cls=None,
        fetch_related_anime=False
    ):
        self.provider = provider
        self.engine = engine
        self.dataset = None
        self.fetch_related_anime = fetch_related_anime
        self.recommendation_model_cls = recommendation_model_cls
        self.profile_model_cls = profile_model_cls

    async def databuilder(self, year, season, user):
        user_profile = None

        if user:
            season_data, user_data, manga_data = await asyncio.gather(
                self.provider.get_seasonal_anime_list(year, season),
                self.provider.get_user_anime_list(user),
                self.provider.get_user_manga_list(user),
            )

            user_profile = self.profile_model_cls(user, user_data, manga_data)
        else:
            season_data = await self.provider.get_seasonal_anime_list(year, season)

        if season_data is not None and self.fetch_related_anime:
            indices = season_data["id"].to_list()
            related_anime = [await self.provider.get_related_anime(index) for index in indices]
            season_data = season_data.with_columns(continuation_to=pl.Series(related_anime))

        data = self.recommendation_model_cls(user_profile, season_data)
        data.nsfw_tags = self.provider.get_nsfw_tags()

        return data

    def get_dataset(self, year, season, user=None):
        return self.databuilder(year, season, user)

    def async_get_dataset(self, year, season, user=None):
        # If we run from jupyter, loop is already running and we need
        # to act differently. If the loop is not running,
        # we break into "normal path" with RuntimeError
        try:
            asyncio.get_running_loop()

            with ThreadPoolExecutor(1) as pool:
                print("using threading")

                return pool.submit(
                    lambda: asyncio.run(self.databuilder(year, season, user))
                ).result()
        except RuntimeError:
            return asyncio.run(self.databuilder(year, season, user))

    def recommend_seasonal_anime(self, year, season, user=None):
        dataset = self.async_get_dataset(year, season, user)

        if user:
            dataset.recommendations = self.engine.fit_predict(dataset)
        else:
            dataset.recommendations = dataset.seasonal.sort("popularity", descending=True)

        return dataset

    def get_categories(self, recommendations):
        return self.engine.categorize_anime(recommendations)
