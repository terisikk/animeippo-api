import asyncio
import functools
import logging
from datetime import timedelta

import aiohttp

from .. import caching as animecache

REQUEST_TIMEOUT = 30
ANI_API_URL = "https://graphql.anilist.co"
HTTP_TOO_MANY_REQUESTS = 429
RATE_LIMIT_WARNING_THRESHOLD = 10

logger = logging.getLogger(__name__)


def rate_limited(func):
    """Decorator that tracks AniList rate limit headers and retries on 429."""

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        response, result = await func(self, *args, **kwargs)

        self.rate_remaining = int(
            response.headers.get("X-RateLimit-Remaining", self.rate_remaining)
        )
        self.rate_limit = int(response.headers.get("X-RateLimit-Limit", self.rate_limit))

        if self.rate_remaining < RATE_LIMIT_WARNING_THRESHOLD:
            logger.warning(f"AniList rate limit low: {self.rate_remaining}/{self.rate_limit}")

        if response.status == HTTP_TOO_MANY_REQUESTS:
            retry_after = int(response.headers.get("Retry-After", 60))
            logger.warning(f"AniList rate limited. Retrying after {retry_after}s")
            await asyncio.sleep(retry_after)
            _, result = await func(self, *args, **kwargs)

        return result

    return wrapper


class AnilistConnection:
    def __init__(self, cache=None):
        self.cache = cache
        self._session = None
        self.rate_remaining = 90
        self.rate_limit = 90

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False

    @animecache.cached_query(ttl=timedelta(days=1))
    async def request_paginated(self, query, parameters):
        anime_list = {"data": {"media": []}}
        variables = parameters.copy()

        async for page in self.get_all_pages(query, variables):
            for item in page["media"]:
                anime_list["data"]["media"].append(item)

        return anime_list

    @animecache.cached_query(ttl=timedelta(days=1))
    async def request_collection(self, query, parameters):
        variables = parameters.copy()
        return await self.request_single(query, variables)

    async def get_session(self):
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=10,
                ttl_dns_cache=300,
            )
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def close(self):
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    @rate_limited
    async def request_single(self, query, variables):
        session = await self.get_session()

        async with session.post(
            ANI_API_URL, json={"query": query, "variables": variables}, timeout=REQUEST_TIMEOUT
        ) as response:
            response.raise_for_status()
            return response, await response.json()

    async def get_all_pages(self, query, variables):
        variables["page"] = 1
        variables["perPage"] = 50

        first_page = await self.request_single(query, variables)
        first_page_data = first_page.get("data", {}).get("Page", None)

        yield first_page_data

        if first_page_data is None:
            return

        page_info = first_page_data.get("pageInfo", {})
        has_next_page = page_info.get("hasNextPage", False)
        last_page = page_info.get("lastPage", 1)

        if not has_next_page or last_page <= 1:
            return

        max_page = min(last_page, 10)

        for page_num in range(2, max_page + 1):
            page_response = await self.request_single(query, {**variables, "page": page_num})
            page_data = page_response.get("data", {}).get("Page", None)
            if page_data is not None:
                yield page_data
