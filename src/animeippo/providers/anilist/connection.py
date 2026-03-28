import asyncio
import functools
import types
from datetime import timedelta

import aiohttp
import structlog

from .. import caching as animecache

REQUEST_TIMEOUT = 30
ANI_API_URL = "https://graphql.anilist.co"
HTTP_BAD_REQUEST = 400
HTTP_TOO_MANY_REQUESTS = 429
RATE_LIMIT_WARNING_THRESHOLD = 10

logger = structlog.get_logger()


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
            logger.warning(
                "rate_limit_low",
                remaining=self.rate_remaining,
                limit=self.rate_limit,
            )

        if response.status == HTTP_TOO_MANY_REQUESTS:
            retry_after = int(response.headers.get("Retry-After", 60))
            logger.warning("rate_limited", retry_after=retry_after)
            await asyncio.sleep(retry_after)
            response, result = await func(self, *args, **kwargs)

        if response.status >= HTTP_BAD_REQUEST:
            logger.error(
                "anilist_api_error",
                status=response.status,
                errors=result.get("errors", "Unknown error"),
                rate_remaining=self.rate_remaining,
                rate_limit=self.rate_limit,
            )
            raise aiohttp.ClientResponseError(
                request_info=aiohttp.RequestInfo(
                    url=ANI_API_URL, method="POST", headers={}, real_url=ANI_API_URL
                ),
                history=(),
                status=response.status,
                message=str(result.get("errors", "Unknown error")),
            )

        return result

    return wrapper


class AnilistConnection:
    def __init__(self, cache=None):
        self.cache = cache
        self.rate_remaining = 90
        self.rate_limit = 90

    @animecache.cached_query(ttl=timedelta(days=1))
    async def request_paginated(self, query, parameters):
        anime_list = {"data": {"media": []}}
        variables = parameters.copy()

        async with aiohttp.ClientSession() as session:
            async for page in self.get_all_pages(session, query, variables):
                for item in page["media"]:
                    anime_list["data"]["media"].append(item)

        return anime_list

    @animecache.cached_query(ttl=timedelta(days=1))
    async def request_collection(self, query, parameters):
        variables = parameters.copy()

        async with aiohttp.ClientSession() as session:
            return await self.request_single(session, query, variables)

    @rate_limited
    async def request_single(self, session, query, variables):
        async with session.post(
            ANI_API_URL, json={"query": query, "variables": variables}, timeout=REQUEST_TIMEOUT
        ) as response:
            body = await response.json()
            info = types.SimpleNamespace(status=response.status, headers=dict(response.headers))
            return info, body

    async def get_all_pages(self, session, query, variables):
        MAX_PAGES = 10

        variables["page"] = 1
        variables["perPage"] = 50

        first_page = await self.request_single(session, query, variables)
        first_page_data = first_page.get("data", {}).get("Page", None)

        yield first_page_data

        if first_page_data is None:
            return

        has_next_page = first_page_data.get("pageInfo", {}).get("hasNextPage", False)
        page_num = 2

        while has_next_page and page_num <= MAX_PAGES:
            logger.debug("fetching_page", page=page_num)
            page_response = await self.request_single(
                session, query, {**variables, "page": page_num}
            )
            page_data = page_response.get("data", {}).get("Page", None)

            if page_data is None:
                return

            yield page_data
            has_next_page = page_data.get("pageInfo", {}).get("hasNextPage", False)
            page_num += 1
