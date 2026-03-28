#!/usr/bin/env python3
"""Pre-load cache with yearly anime and static data.

This script should be run nightly (via cron) to:
1. Warm the cache with previous, current, and next year anime (full years)
2. Cache Genre/Tag collections from AniList API in Redis
3. Pre-cache data for better user experience

Usage:
    python scripts/preload_cache.py [--skip-static]
"""

import argparse
import asyncio
from datetime import datetime, timedelta

import aiohttp
import dotenv
import structlog

from animeippo.logging import configure_logging

dotenv.load_dotenv("conf/prod.env")
configure_logging()
logger = structlog.get_logger()


async def get_years_to_preload():
    current_year = datetime.now().year
    return [current_year - 1, current_year, current_year + 1]


async def preload_year_anime(provider, year):
    """Pre-load all anime for a full year (no season specified)."""
    logger.info("preload_year_start", year=year)
    try:
        anime_list = await provider.get_seasonal_anime_list(str(year), None)
    except Exception:
        logger.error("preload_year_error", year=year, exc_info=True)
        return 0

    if anime_list is not None:
        count = len(anime_list)
        logger.info("preload_year_done", year=year, count=count)
        return count

    logger.warning("preload_year_empty", year=year)
    return 0


async def fetch_and_cache_genres(connection, cache):
    """Fetch all genres from AniList API and cache them."""
    query = """
    query {
        GenreCollection
    }
    """

    try:
        async with aiohttp.ClientSession() as session:
            response = await connection.request_single(session, query, {})
    except Exception:
        logger.error("genres_fetch_error", exc_info=True)
        return None

    genres = response.get("data", {}).get("GenreCollection", [])

    if cache and cache.is_available():
        cache.set_json("anilist:genres", genres, ttl=timedelta(days=30))
        logger.info("genres_cached", count=len(genres))
    else:
        logger.warning("genres_fetched_no_cache", count=len(genres))

    return genres


async def fetch_and_cache_tags(connection, cache):
    """Fetch all media tags from AniList API and cache them."""
    query = """
    query {
        MediaTagCollection {
            id
            name
            category
            isAdult
        }
    }
    """

    try:
        async with aiohttp.ClientSession() as session:
            response = await connection.request_single(session, query, {})
    except Exception:
        logger.error("tags_fetch_error", exc_info=True)
        return None

    tags_data = response.get("data", {}).get("MediaTagCollection", [])

    if cache and cache.is_available():
        cache.set_json("anilist:tags", tags_data, ttl=timedelta(days=30))

        tag_lookup = {
            tag["id"]: {
                "name": tag["name"],
                "category": tag["category"],
                "isAdult": tag.get("isAdult", False),
            }
            for tag in tags_data
        }
        cache.set_json("anilist:tag_lookup", tag_lookup, ttl=timedelta(days=30))

        nsfw_tags = [tag["name"] for tag in tags_data if tag.get("isAdult", False)]
        cache.set_json("anilist:nsfw_tags", nsfw_tags, ttl=timedelta(days=30))

        logger.info("tags_cached", total=len(tags_data), nsfw=len(nsfw_tags))
    else:
        logger.warning("tags_fetched_no_cache", total=len(tags_data))

    return tags_data


async def main():
    """Main pre-loading routine."""
    parser = argparse.ArgumentParser(description="Pre-load cache with yearly anime data")
    parser.add_argument(
        "--skip-static",
        action="store_true",
        help="Skip fetching genre/tag collections",
    )
    args = parser.parse_args()

    from animeippo.cache import RedisCache
    from animeippo.providers.anilist import AniListProvider

    cache = RedisCache()

    if not cache.is_available():
        logger.error("redis_unavailable")
        return 1

    provider = AniListProvider(cache=cache)

    logger.info("preload_start")
    years_to_load = await get_years_to_preload()
    logger.info("preload_years", years=years_to_load)

    total_cached = 0
    for year in years_to_load:
        count = await preload_year_anime(provider, year)
        total_cached += count

    logger.info("preload_years_done", total=total_cached)

    if not args.skip_static:
        logger.info("preload_static_start")
        await fetch_and_cache_genres(provider.connection, cache)
        await fetch_and_cache_tags(provider.connection, cache)

    logger.info("preload_complete")
    return 0


def cli():
    """CLI entry point for the preload cache script."""
    import sys

    exit_code = asyncio.run(main())
    sys.exit(exit_code)


if __name__ == "__main__":
    cli()
