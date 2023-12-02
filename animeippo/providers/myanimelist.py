import dotenv
import os
import aiohttp

from datetime import timedelta

from . import provider
from .formatters import mal_formatter

import animeippo.cache as animecache

dotenv.load_dotenv("conf/prod.env")

MAL_API_URL = "https://api.myanimelist.net/v2"
MAL_API_TOKEN = os.environ.get("MAL_API_TOKEN", None)

HEADERS = {"Authorization": f"Bearer {MAL_API_TOKEN}"}
REQUEST_TIMEOUT = 30


class MyAnimeListProvider(provider.AbstractAnimeProvider):
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

        return mal_formatter.transform_watchlist_data(anime_list, self.get_feature_fields())

    @animecache.cached_dataframe(ttl=timedelta(days=1))
    async def get_seasonal_anime_list(self, year, season):
        if not year or not season:
            return None

        query = f"/anime/season/{year}/{season}"
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

        anime_list = await self.connection.request_anime_list(query, parameters)

        return mal_formatter.transform_seasonal_data(anime_list, self.get_feature_fields())

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

        return mal_formatter.transform_related_anime(anime_list, self.get_feature_fields())

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

        return mal_formatter.transform_user_manga_list_data(anime_list, self.get_feature_fields())

    def get_feature_fields(self):
        return ["genres", "rating"]


class MyAnimeListConnection:
    def __init__(self, cache=None):
        self.cache = cache

    @animecache.cached_query(ttl=timedelta(days=1))
    async def request_anime_list(self, query, parameters):
        anime_list = {"data": []}

        async with aiohttp.ClientSession() as session:
            async for page in self.requests_get_all_pages(session, query, parameters):
                for item in page["data"]:
                    anime_list["data"].append(item)

        return anime_list

    @animecache.cached_query(ttl=timedelta(days=7))
    async def request_related_anime(self, query, parameters):
        anime = {"data": []}

        async with aiohttp.ClientSession() as session:
            async with session.get(
                MAL_API_URL + query,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
                params=parameters,
            ) as response:
                response.raise_for_status()
                details = await response.json()
                anime["data"] = details["related_anime"]

        return anime

    async def requests_get_next_page(self, session, page):
        if page:
            next_page = None
            next_page_url = page.get("paging", dict()).get("next", None)

            if next_page_url:
                async with session.get(
                    next_page_url,
                    headers=HEADERS,
                    timeout=REQUEST_TIMEOUT,
                ) as response:
                    response.raise_for_status()
                    next_page = await response.json()
                    return next_page

    async def requests_get_all_pages(self, session, query, parameters):
        async with session.get(
            MAL_API_URL + query,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
            params=parameters,
        ) as response:
            response.raise_for_status()
            page = await response.json()

            while page:
                yield page
                page = await self.requests_get_next_page(session, page)
