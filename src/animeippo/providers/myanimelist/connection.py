import asyncio
import os
from datetime import timedelta

import aiohttp
import dotenv
import structlog

from .. import caching as animecache

MAL_API_URL = "https://api.myanimelist.net/v2"
MAL_AUTH_URL = "https://myanimelist.net/v1/oauth2/token"
ENV_FILE = "conf/prod.env"
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=60)
MAL_REQUEST_INTERVAL = 1  # seconds between requests
HTTP_UNAUTHORIZED = 401

logger = structlog.get_logger()


class MyAnimeListConnection:
    def __init__(self, cache=None):
        self.cache = cache
        self.access_token = os.environ.get("MAL_API_TOKEN", None)
        self.refresh_token = os.environ.get("MAL_REFRESH_TOKEN", None)
        self.client_id = os.environ.get("MAL_CLIENT_ID", None)
        self.client_secret = os.environ.get("MAL_CLIENT_SECRET", None)

    @property
    def headers(self):
        return {"Authorization": f"Bearer {self.access_token}"}

    @animecache.cached_query(ttl=timedelta(days=1))
    async def request_anime_list(self, query, parameters):
        anime_list = {"data": []}

        async with aiohttp.ClientSession() as session:
            async for page in self.requests_get_all_pages(session, query, parameters):
                for item in page["data"]:
                    anime_list["data"].append(item)

        return anime_list

    async def request_with_retry(self, session, method, url, **kwargs):
        """Make a request, refreshing the token on 401."""
        kwargs["headers"] = self.headers
        kwargs["timeout"] = REQUEST_TIMEOUT

        async with session.request(method, url, **kwargs) as response:
            if response.status == HTTP_UNAUTHORIZED and self.refresh_token:
                logger.warning("mal_token_expired")
                await self.do_token_refresh()
                kwargs["headers"] = self.headers
                async with session.request(method, url, **kwargs) as retry_response:
                    retry_response.raise_for_status()
                    return await retry_response.json()

            response.raise_for_status()
            return await response.json()

    async def do_token_refresh(self):
        """Refresh the MAL access token using the refresh token."""
        async with aiohttp.ClientSession() as session:
            data = {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
            }

            async with session.post(MAL_AUTH_URL, data=data, timeout=REQUEST_TIMEOUT) as response:
                response.raise_for_status()
                tokens = await response.json()

                self.access_token = tokens["access_token"]
                self.refresh_token = tokens.get("refresh_token", self.refresh_token)

                self.persist_tokens()
                logger.info("mal_token_refreshed")

    def persist_tokens(self):
        """Write refreshed tokens to env file so they survive restarts."""
        dotenv.set_key(ENV_FILE, "MAL_API_TOKEN", self.access_token)
        dotenv.set_key(ENV_FILE, "MAL_REFRESH_TOKEN", self.refresh_token)

    async def requests_get_next_page(self, session, page):
        if page:
            next_page_url = page.get("paging", {}).get("next", None)

            if next_page_url:
                await asyncio.sleep(MAL_REQUEST_INTERVAL)
                return await self.request_with_retry(session, "GET", next_page_url)

        return None

    async def requests_get_all_pages(self, session, query, parameters):
        result = await self.request_with_retry(
            session, "GET", MAL_API_URL + query, params=parameters
        )

        page = result
        while page:
            yield page
            page = await self.requests_get_next_page(session, page)
