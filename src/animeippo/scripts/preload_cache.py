#!/usr/bin/env python3
"""Pre-load cache with yearly anime and check tag freshness.

This script should be run nightly (via cron) to:
1. Warm the cache with previous, current, and next year anime (full years)
2. Compare AniList API tags against static data and log new/removed tags

Usage:
    python scripts/preload_cache.py [--skip-static]
"""

import argparse
import asyncio
from datetime import datetime

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
        logger.exception("preload_year_error", year=year)
        return 0

    if anime_list is not None:
        count = len(anime_list)
        logger.info("preload_year_done", year=year, count=count)
        return count

    logger.warning("preload_year_empty", year=year)
    return 0


async def check_tag_freshness(connection):
    """Compare AniList API tags against static data and log differences."""
    from animeippo.providers.anilist import data

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
        logger.exception("tags_fetch_error")
        return

    api_tags = response.get("data", {}).get("MediaTagCollection", [])
    api_ids = {tag["id"] for tag in api_tags}
    static_ids = set(data.ALL_TAGS.keys())

    new_tags = [tag for tag in api_tags if tag["id"] not in static_ids]
    removed_ids = static_ids - api_ids

    if new_tags:
        logger.warning(
            "new_tags_on_anilist",
            count=len(new_tags),
            tags=[t["name"] for t in new_tags],
        )
    if removed_ids:
        logger.warning(
            "removed_tags_on_anilist",
            count=len(removed_ids),
            ids=sorted(removed_ids),
        )
    if not new_tags and not removed_ids:
        logger.info("tags_up_to_date", count=len(api_tags))


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
        logger.info("preload_static_check")
        await check_tag_freshness(provider.connection)

    logger.info("preload_complete")
    return 0


def cli():
    """CLI entry point for the preload cache script."""
    import sys

    exit_code = asyncio.run(main())
    sys.exit(exit_code)


if __name__ == "__main__":
    cli()
