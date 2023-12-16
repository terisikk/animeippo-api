import aiohttp
from datetime import timedelta
from async_lru import alru_cache

from . import provider
from .formatters import ani_formatter

import animeippo.cache as animecache

REQUEST_TIMEOUT = 30
ANI_API_URL = "https://graphql.anilist.co"


class AniListProvider(provider.AbstractAnimeProvider):
    def __init__(self, cache=None):
        self.cache = cache
        self.connection = AnilistConnection(cache)

    @alru_cache(maxsize=1)
    @animecache.cached_dataframe(ttl=timedelta(days=1))
    async def get_user_anime_list(self, user_id):
        if user_id is None:
            return None

        anime_list = {"data": []}

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

        variables = {"userName": user_id}

        collection = await self.connection.request_collection(query, variables)

        for coll in collection["data"]["MediaListCollection"]["lists"]:
            for entry in coll["entries"]:
                anime_list["data"].append(entry)

        return ani_formatter.transform_watchlist_data(anime_list, self.get_feature_fields())

    @animecache.cached_dataframe(ttl=timedelta(days=1))
    async def get_seasonal_anime_list(self, year, season):
        if year is None:
            return None

        query = ""
        variables = {}

        if season is not None:
            query = """
            query ($seasonYear: Int, $season: MediaSeason, $page: Int) {
                Page(page: $page, perPage: 50) {
                    pageInfo { hasNextPage currentPage lastPage total perPage }
                    media(seasonYear: $seasonYear, season: $season, type:ANIME) {
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

            variables = {"seasonYear": int(year), "season": str(season).upper()}
        else:
            query = """
            query ($seasonYear: Int, $page: Int) {
                Page(page: $page, perPage: 50) {
                    pageInfo { hasNextPage currentPage lastPage total perPage }
                    media(seasonYear: $seasonYear, type:ANIME) {
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

            variables = {"seasonYear": int(year)}

        anime_list = await self.connection.request_paginated(query, variables)

        return ani_formatter.transform_seasonal_data(anime_list, self.get_feature_fields())

    @alru_cache(maxsize=1)
    # @animecache.cached_dataframe(ttl=timedelta(days=1))
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

        return ani_formatter.transform_user_manga_list_data(manga_list, self.get_feature_fields())

    def get_feature_fields(self):
        return ["genres", "tags"]

    def get_related_anime(self, id):
        pass


class AnilistConnection:
    def __init__(self, cache=None):
        self.cache = cache

    @animecache.cached_query(ttl=timedelta(days=1))
    async def request_paginated(self, query, parameters):
        anime_list = {"data": {"media": []}}
        variables = parameters.copy()  # To avoid cache miss with side effects

        async for page in self.requests_get_all_pages(query, variables):
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

    async def requests_get_all_pages(self, query, variables):
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
