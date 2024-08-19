import os
from datetime import timedelta

from async_lru import alru_cache

from animeippo.providers.anilist.connection import AnilistConnection

from .. import abstract_provider
from .. import caching as animecache
from . import data, formatter

USER_DATA_TTL_DAYS = os.environ.get("USER_DATA_TTL_DAYS", 1)
SEASONAL_DATA_TTL_DAYS = os.environ.get("SEASONAL_DATA_TTL_DAYS", 7)


class AniListProvider(abstract_provider.AbstractAnimeProvider):
    def __init__(self, cache=None):
        self.cache = cache
        self.connection = AnilistConnection(cache)

    @alru_cache(maxsize=1)
    @animecache.cached_dataframe(ttl=timedelta(days=USER_DATA_TTL_DAYS))
    async def get_user_anime_list(self, user_id):
        if user_id is None:
            return None

        anime_list = {"data": []}

        # fmt: off
        query = """
        query ($userName: String) {
            MediaListCollection(userName: $userName, type: ANIME) {
                lists {
                    name
                    status
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
                                name
                                rank
                                isAdult
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
            for entry in coll["entries"]:
                anime_list["data"].append(entry)

        return formatter.transform_watchlist_data(anime_list, self.get_feature_fields())

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
                                name
                                rank
                                isAdult
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
                                name
                                rank
                                isAdult
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

        return formatter.transform_seasonal_data(anime_list, self.get_feature_fields())

    @alru_cache(maxsize=1)
    @animecache.cached_dataframe(ttl=timedelta(days=USER_DATA_TTL_DAYS))
    async def get_user_manga_list(self, user_id):
        if user_id is None:
            return None

        manga_list = {"data": []}

        query = """
        query ($userName: String) {
            MediaListCollection(userName: $userName, type: MANGA) {
                lists {
                    name
                    status
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
                                name
                                rank
                                isAdult
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
            for entry in coll["entries"]:
                manga_list["data"].append(entry)

        return formatter.transform_user_manga_list_data(manga_list, self.get_feature_fields())

    def get_feature_fields(self):
        return ["genres", "tags"]

    def get_related_anime(self, related_id):
        pass

    def get_nsfw_tags(self):
        return data.NSFW_TAGS

    def get_genres(self):
        return data.ALL_GENRES
