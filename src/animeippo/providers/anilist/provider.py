import os
from datetime import timedelta

from asyncache import cached as async_cached
from cachetools import TTLCache

from animeippo.providers.anilist.connection import AnilistConnection

from .. import abstract_provider
from .. import caching as animecache
from . import data, formatter

USER_DATA_TTL_DAYS = int(os.environ.get("USER_DATA_TTL_DAYS", "1"))
SEASONAL_DATA_TTL_DAYS = int(os.environ.get("SEASONAL_DATA_TTL_DAYS", "7"))


class AniListProvider(abstract_provider.AbstractAnimeProvider):
    def __init__(self, cache=None):
        self.cache = cache
        self.connection = AnilistConnection(cache)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connection.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - ensures connection is properly closed."""
        return await self.connection.__aexit__(exc_type, exc_val, exc_tb)

    @async_cached(
        TTLCache(
            maxsize=1,
            ttl=timedelta(days=USER_DATA_TTL_DAYS).total_seconds(),
        )
    )
    @animecache.cached_dataframe(ttl=timedelta(days=USER_DATA_TTL_DAYS))
    async def get_user_anime_list(self, user_id):
        if user_id is None:
            return None

        anime_list = {"data": []}

        # fmt: off
        query = """
        query ($userName: String) {
            MediaListCollection(userName: $userName, type: ANIME, forceSingleCompletedList: true) {
                lists {
                    name
                    status
                    isCustomList
                    entries {
                        status
                        score(format:POINT_10)
                        completedAt {
                            year
                            month
                            day
                        }
                        media {
                            id
                            idMal
                            title { romaji }
                            format
                            genres
                            tags {
                                id
                                rank
                            }
                            meanScore
                            duration
                            episodes
                            source
                            studios { edges { node { name isAnimationStudio } }}
                            seasonYear
                            season
                            coverImage { large }
                            staff { edges {role} nodes {id}}
                        }
                    }
                }
            }
        }
        """
        # fmt: on

        variables = {"userName": user_id}

        collection = await self.connection.request_collection(query, variables)

        for coll in collection["data"]["MediaListCollection"]["lists"]:
            if not coll.get("isCustomList", False):
                for entry in coll["entries"]:
                    anime_list["data"].append(entry)

        return formatter.transform_watchlist_data(
            anime_list, self.get_feature_fields(), self.get_tag_lookup()
        )

    @async_cached(
        TTLCache(
            maxsize=1,
            ttl=timedelta(days=SEASONAL_DATA_TTL_DAYS).total_seconds(),
        )
    )
    @animecache.cached_dataframe(ttl=timedelta(days=SEASONAL_DATA_TTL_DAYS))
    async def get_seasonal_anime_list(self, year, season):
        if year is None:
            return None

        query = ""
        variables = {}

        if season is not None:
            # fmt: off
            query = """
            query ($seasonYear: Int, $season: MediaSeason, $page: Int) {
                Page(page: $page, perPage: 50) {
                    pageInfo { hasNextPage currentPage lastPage total perPage }
                    media(seasonYear: $seasonYear, season: $season, type: ANIME,
                        isAdult: false, tag_not_in: ["Kids"]) {
                            id
                            idMal
                            title { romaji }
                            status
                            format
                            genres
                            tags {
                                id
                                rank
                            }
                            meanScore
                            duration
                            episodes
                            source
                            studios { edges { node { name isAnimationStudio } }}
                            seasonYear
                            season
                            relations { edges { relationType, node { id, idMal }}}
                            popularity
                            coverImage { large }
                            staff { edges {role} nodes {id}}
                    }
                }
            }
            """
            # fmt: on

            variables = {"seasonYear": int(year), "season": str(season).upper()}
        else:
            # fmt: off
            query = """
            query ($seasonYear: Int, $page: Int) {
                Page(page: $page, perPage: 50) {
                    pageInfo { hasNextPage currentPage lastPage total perPage }
                    media(seasonYear: $seasonYear, type: ANIME, isAdult: false,
                        tag_not_in: ["Kids"]) {
                            id
                            idMal
                            title { romaji }
                            status
                            format
                            genres
                            tags {
                                id
                                rank
                            }
                            meanScore
                            duration
                            episodes
                            source
                            studios { edges { node { name isAnimationStudio } }}
                            seasonYear
                            season
                            relations { edges { relationType, node { id, idMal }}}
                            popularity
                            coverImage { large }
                            staff { edges {role} nodes {id}}
                    }
                }
            }
            """
            # fmt: on

            variables = {"seasonYear": int(year)}

        anime_list = await self.connection.request_paginated(query, variables)

        return formatter.transform_seasonal_data(
            anime_list, self.get_feature_fields(), self.get_tag_lookup()
        )

    @async_cached(
        TTLCache(
            maxsize=1,
            ttl=timedelta(days=USER_DATA_TTL_DAYS).total_seconds(),
        )
    )
    @animecache.cached_dataframe(ttl=timedelta(days=USER_DATA_TTL_DAYS))
    async def get_user_manga_list(self, user_id):
        if user_id is None:
            return None

        manga_list = {"data": []}

        query = """
        query ($userName: String) {
            MediaListCollection(userName: $userName, type: MANGA, forceSingleCompletedList: true) {
                lists {
                    name
                    status
                    isCustomList
                    entries {
                        status
                        score(format:POINT_10)
                        completedAt {
                            year
                            month
                            day
                        }
                        media {
                            id
                            idMal
                            title { romaji }
                            genres
                            tags {
                                id
                                rank
                            }
                            meanScore
                        }
                    }
                }
            }
        }
        """

        variables = {"userName": user_id}

        collection = await self.connection.request_collection(query, variables)

        for coll in collection["data"]["MediaListCollection"]["lists"]:
            if not coll.get("isCustomList", False):
                for entry in coll["entries"]:
                    manga_list["data"].append(entry)

        return formatter.transform_user_manga_list_data(
            manga_list, self.get_feature_fields(), self.get_tag_lookup()
        )

    def get_feature_fields(self):
        return ["genres", "tags"]

    def get_related_anime(self, related_id):
        pass

    def get_nsfw_tags(self):
        """Get NSFW tags from cache if available, otherwise use static data."""
        if self.cache and self.cache.is_available():
            cached_tags = self.cache.get_json("anilist:nsfw_tags")
            if cached_tags:
                return set(cached_tags)

        return data.NSFW_TAGS

    def get_genres(self):
        """Get genres from cache if available, otherwise use static data."""
        if self.cache and self.cache.is_available():
            cached_genres = self.cache.get_json("anilist:genres")
            if cached_genres:
                return set(cached_genres)

        return data.ALL_GENRES

    def get_tag_lookup(self):
        """Get tag lookup dict (id -> {name, category, isAdult}) from cache or static data."""
        if self.cache and self.cache.is_available():
            tag_lookup = self.cache.get_json("anilist:tag_lookup")
            if tag_lookup:
                return {int(k): v for k, v in tag_lookup.items()}

        return data.ALL_TAGS
