import os
from datetime import timedelta

import aiohttp

from .. import caching as animecache

MAL_API_URL = "https://api.myanimelist.net/v2"
MAL_API_TOKEN = os.environ.get("MAL_API_TOKEN", None)
HEADERS = {"Authorization": f"Bearer {MAL_API_TOKEN}"}
REQUEST_TIMEOUT = 30


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
            next_page_url = page.get("paging", {}).get("next", None)

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
