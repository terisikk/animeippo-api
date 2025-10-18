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

import dotenv

dotenv.load_dotenv("conf/prod.env")


async def get_years_to_preload():
    """Get previous year, current year, and next year for pre-loading.

    Returns list of years to pre-load (full year, no season specified).
    """
    current_year = datetime.now().year
    return [current_year - 1, current_year, current_year + 1]


async def preload_year_anime(provider, year):
    """Pre-load all anime for a full year (no season specified)."""
    print(f"Pre-loading year {year} anime...")
    try:
        # Pass None for season to get all anime from that year
        anime_list = await provider.get_seasonal_anime_list(str(year), None)
        if anime_list is not None:
            count = len(anime_list)
            print(f"  ✓ Cached {count} anime for year {year}")
            return count
        print(f"  ✗ No data returned for year {year}")
        return 0
    except Exception as e:
        print(f"  ✗ Error pre-loading year {year}: {e}")
        return 0


async def fetch_and_cache_genres(connection, cache):
    """Fetch all genres from AniList API and cache them."""
    query = """
    query {
        GenreCollection
    }
    """

    try:
        response = await connection.request_single(query, {})
        genres = response.get("data", {}).get("GenreCollection", [])

        if cache and cache.is_available():
            # Cache for 30 days (genres change very rarely)
            cache.set_json("anilist:genres", genres, ttl=timedelta(days=30))
            print(f"  ✓ Fetched and cached {len(genres)} genres")
        else:
            print(f"  ✓ Fetched {len(genres)} genres (cache unavailable)")

        return genres
    except Exception as e:
        print(f"  ✗ Error fetching genres: {e}")
        return None


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
        response = await connection.request_single(query, {})
        tags_data = response.get("data", {}).get("MediaTagCollection", [])

        if cache and cache.is_available():
            # Cache full tag details for 30 days (tags change rarely)
            cache.set_json("anilist:tags", tags_data, ttl=timedelta(days=30))

            # Create tag lookup by ID for fast enrichment (id -> {name, category, isAdult})
            tag_lookup = {
                tag["id"]: {
                    "name": tag["name"],
                    "category": tag["category"],
                    "isAdult": tag.get("isAdult", False),
                }
                for tag in tags_data
            }
            cache.set_json("anilist:tag_lookup", tag_lookup, ttl=timedelta(days=30))

            # Also create separate NSFW tags list for convenience
            nsfw_tags = [tag["name"] for tag in tags_data if tag.get("isAdult", False)]
            cache.set_json("anilist:nsfw_tags", nsfw_tags, ttl=timedelta(days=30))

            print(f"  ✓ Fetched and cached {len(tags_data)} tags ({len(nsfw_tags)} NSFW)")
        else:
            print(f"  ✓ Fetched {len(tags_data)} tags (cache unavailable)")

        return tags_data
    except Exception as e:
        print(f"  ✗ Error fetching tags: {e}")
        return None


async def main():
    """Main pre-loading routine."""
    parser = argparse.ArgumentParser(description="Pre-load cache with yearly anime data")
    parser.add_argument(
        "--skip-static",
        action="store_true",
        help="Skip fetching genre/tag collections",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("AniList Cache Pre-loading Script")
    print("=" * 60)

    from animeippo.cache import RedisCache
    from animeippo.providers.anilist import AniListProvider

    # Initialize provider with cache
    cache = RedisCache()

    if not cache.is_available():
        print("\n❌ ERROR: Redis cache is not available!")
        print("Pre-loading requires Redis to be running.")
        print("Please start Redis and try again.")
        print("=" * 60)
        return 1

    async with AniListProvider(cache=cache) as provider:
        # Pre-load previous, current, and next year anime
        print("\n1. Pre-loading anime by year...")
        years_to_load = await get_years_to_preload()
        print(f"Years to pre-load: {', '.join(map(str, years_to_load))}")

        total_cached = 0
        for year in years_to_load:
            count = await preload_year_anime(provider, year)
            total_cached += count

        print(f"\nTotal anime cached: {total_cached}")

        # Fetch and cache genre/tag collections
        if not args.skip_static:
            print("\n2. Fetching and caching Genre collection...")
            await fetch_and_cache_genres(provider.connection, cache)

            print("\n3. Fetching and caching Tag collection...")
            await fetch_and_cache_tags(provider.connection, cache)

    print("\n" + "=" * 60)
    print("Pre-loading complete!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    import sys

    exit_code = asyncio.run(main())
    sys.exit(exit_code)
