import asyncio
from datetime import timedelta

import aiohttp

from .. import caching as animecache

REQUEST_TIMEOUT = 30
ANI_API_URL = "https://graphql.anilist.co"


class AnilistConnection:
    def __init__(self, cache=None):
        self.cache = cache
        self._session = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - ensures session is properly closed."""
        await self.close()
        return False

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

    async def get_session(self):
        """Get or create a persistent aiohttp ClientSession for connection reuse."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close the persistent session. Should be called when done with the connection."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    async def request_single(self, query, variables):
        url = ANI_API_URL
        session = await self.get_session()

        async with session.post(
            url, json={"query": query, "variables": variables}, timeout=REQUEST_TIMEOUT
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def get_all_pages(self, query, variables):
        variables["page"] = 1
        variables["perPage"] = 50

        # Fetch first page to determine total pages
        first_page = await self.request_single(query, variables)
        first_page_data = first_page.get("data", {}).get("Page", None)

        yield first_page_data

        if first_page_data is None:
            return

        # Check if there are more pages
        page_info = first_page_data.get("pageInfo", {})
        has_next_page = page_info.get("hasNextPage", False)
        last_page = page_info.get("lastPage", 1)

        if not has_next_page or last_page <= 1:
            return

        # Safeguard: limit to 10 pages max
        max_page = min(last_page, 10)

        # Fetch remaining pages in parallel
        remaining_page_numbers = range(2, max_page + 1)
        remaining_page_tasks = [
            self.request_single(query, {**variables, "page": page_num})
            for page_num in remaining_page_numbers
        ]

        remaining_pages = await asyncio.gather(*remaining_page_tasks)

        for page_response in remaining_pages:
            page_data = page_response.get("data", {}).get("Page", None)
            if page_data is not None:
                yield page_data
