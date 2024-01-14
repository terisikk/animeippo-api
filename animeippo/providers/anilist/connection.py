from .. import caching as animecache

import aiohttp

from datetime import timedelta

REQUEST_TIMEOUT = 30
ANI_API_URL = "https://graphql.anilist.co"


class AnilistConnection:
    def __init__(self, cache=None):
        self.cache = cache

    @animecache.cached_query(ttl=timedelta(days=1))
    async def request_paginated(self, query, parameters):
        anime_list = {"data": {"media": []}}
        variables = parameters.copy()  # To avoid cache miss with side effects

        async for page in self.get_all_pages(query, variables):
            for item in page["media"]:
                anime_list["data"]["media"].append(item)

        return anime_list

    @animecache.cached_query(ttl=timedelta(days=1))
    async def request_collection(self, query, parameters):
        variables = parameters.copy()  # To avoid cache miss with side effects

        return await self.request_single(query, variables)

    async def request_single(self, query, variables):
        url = ANI_API_URL

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json={"query": query, "variables": variables}, timeout=REQUEST_TIMEOUT
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def get_all_pages(self, query, variables):
        variables["page"] = 0
        variables["perPage"] = 50

        page = await self.request_single(query, variables)
        page = page.get("data", {}).get("Page", None)

        safeguard = 10

        yield page

        while page is not None and page["pageInfo"].get("hasNextPage", False) and safeguard > 0:
            variables["page"] = page["pageInfo"]["currentPage"] + 1

            page = await self.request_single(query, variables)
            page = page["data"]["Page"]
            yield page
            safeguard = safeguard - 1
