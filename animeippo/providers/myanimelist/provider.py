from datetime import timedelta

import polars as pl

from animeippo.providers.myanimelist.connection import MyAnimeListConnection

from .. import abstract_provider, caching as animecache
from . import formatter


class MyAnimeListProvider(abstract_provider.AbstractAnimeProvider):
    def __init__(self, cache=None):
        self.connection = MyAnimeListConnection(cache)
        self.cache = cache

    @animecache.cached_dataframe(ttl=timedelta(days=1))
    async def get_user_anime_list(self, user_id):
        if not user_id:
            return None

        query = f"/users/{user_id}/animelist"
        fields = [
            "id",
            "title",
            "media_type",
            "genres",
            "list_status{score,status,finish_date}",
            "status",
            "studios",
            "rating{value}",
            "mean",
            "num_episodes",
            "start_season",
            "source",
            "main_picture{medium}",
            "average_episode_duration",
        ]

        parameters = {"nsfw": "true", "fields": ",".join(fields)}

        anime_list = await self.connection.request_anime_list(query, parameters)

        return formatter.transform_watchlist_data(anime_list, self.get_feature_fields())

    @animecache.cached_dataframe(ttl=timedelta(days=1))
    async def get_seasonal_anime_list(self, year, season):
        if not year:
            return None

        fields = [
            "id",
            "title",
            "media_type",
            "genres",
            "status",
            "studios",
            "num_list_users",
            "num_episodes",
            "rating{value}",
            "mean",
            "start_season",
            "source",
            "main_picture{medium}",
            "average_episode_duration",
        ]
        parameters = {"nsfw": "true", "fields": ",".join(fields)}

        if season is not None:
            query = f"/anime/season/{year}/{season}"

            anime_list = await self.connection.request_anime_list(query, parameters)

            transformed = formatter.transform_seasonal_data(anime_list, self.get_feature_fields())

            return transformed.filter(
                (~pl.col("rating").is_in(["g", "rx"]))
                & (pl.col("season") == season)
                & (pl.col("season_year") == int(year))
            )

        else:
            responses = []

            for season in ["winter", "spring", "summer", "fall"]:
                query = f"/anime/season/{year}/{season}"
                anime_list = await self.connection.request_anime_list(query, parameters)

                responses.append(
                    formatter.transform_seasonal_data(anime_list, self.get_feature_fields())
                )
            combined = formatter.combine_dataframes(responses)

            return combined.filter(
                (~pl.col("rating").is_in(["g", "rx"])) & (pl.col("season_year") == int(year))
            )

    async def get_related_anime(self, anime_id):
        if not anime_id:
            return None

        query = f"/anime/{anime_id}"

        fields = [
            "id",
            "related_anime",
        ]
        parameters = {"fields": ",".join(fields)}

        anime_list = await self.connection.request_related_anime(query, parameters)

        return formatter.transform_related_anime(anime_list, self.get_feature_fields())

    @animecache.cached_dataframe(ttl=timedelta(days=1))
    async def get_user_manga_list(self, user_id):
        if not user_id:
            return None

        query = f"/users/{user_id}/mangalist"
        fields = [
            "id",
            "title",
            "media_type",
            "genres",
            "list_status{score,status,finish_date}",
            "status",
            "authors",
            "rating{value}",
            "mean",
            "source",
            "main_picture{medium}",
        ]

        parameters = {"nsfw": "true", "fields": ",".join(fields)}

        anime_list = await self.connection.request_anime_list(query, parameters)

        return formatter.transform_user_manga_list_data(anime_list, self.get_feature_fields())

    def get_feature_fields(self):
        return ["genres", "rating"]

    def get_nsfw_tags(self):
        return []
