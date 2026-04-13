from datetime import timedelta

import aiohttp

from .. import abstract_provider
from .. import caching as animecache
from ..anilist import provider as ani
from ..myanimelist.connection import MyAnimeListConnection
from . import formatter

ANILIST_ID_BATCH_SIZE = 50


class MixedProvider(abstract_provider.AbstractAnimeProvider):
    def __init__(self, cache=None):
        self.cache = cache

        self.ani_provider = ani.AniListProvider(cache)
        self.mal_connection = MyAnimeListConnection(cache)

    @animecache.cached_dataframe(ttl=timedelta(days=1))
    async def get_user_anime_list(self, user_id):
        if not user_id:
            return None

        mal_query = f"/users/{user_id}/animelist"
        fields = [
            "id",
            "list_status{score,status,finish_date}",
        ]

        parameters = {"nsfw": "true", "fields": ",".join(fields), "limit": "1000"}

        mal_list = await self.mal_connection.request_anime_list(mal_query, parameters)
        mal_df = formatter.transform_mal_watchlist_data(mal_list)

        ani_query = """
        query ($idMal_in: [Int], $page: Int) {
            Page(page: $page, perPage: 50) {
                pageInfo { hasNextPage currentPage lastPage total perPage }
                media(idMal_in: $idMal_in, type:ANIME) {
                    id
                    idMal
                    title { romaji }
                    format
                    genres
                    tags {
                        name
                        rank
                        isAdult
                        category
                    }
                    meanScore
                    duration
                    episodes
                    source
                    studios { edges { node { name isAnimationStudio } }}
                    seasonYear
                    season
                    coverImage { large }
                    relations { edges { relationType, node { id, idMal }}}
                }
            }
        }
        """

        ani_list = await self.request_anilist_batched(ani_query, mal_df["id"].to_list())

        return formatter.transform_ani_watchlist_data(ani_list, mal_df)

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
                    media(seasonYear: $seasonYear, season: $season, type:ANIME,
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
                            category
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
                    media(seasonYear: $seasonYear, type:ANIME,
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
                            category
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
                    }
                }
            }
            """

            variables = {"seasonYear": int(year), "season": str(season).upper()}

        anime_list = await self.ani_provider.connection.request_paginated(query, variables)

        return formatter.transform_ani_seasonal_data(anime_list)

    @animecache.cached_dataframe(ttl=timedelta(days=1))
    async def get_user_manga_list(self, user_id):
        if user_id is None:
            return None

        mal_query = f"/users/{user_id}/mangalist"
        fields = [
            "id",
            "list_status{score,status}",
        ]

        parameters = {"nsfw": "true", "fields": ",".join(fields), "limit": "1000"}

        mal_list = await self.mal_connection.request_anime_list(mal_query, parameters)
        mal_df = formatter.transform_mal_manga_data(mal_list)

        ani_query = """
        query ($idMal_in: [Int], $page: Int) {
            Page(page: $page, perPage: 50) {
                pageInfo { hasNextPage currentPage lastPage total perPage }
                media(idMal_in: $idMal_in, type:MANGA) {
                    id
                    idMal
                    title { romaji }
                    genres
                    tags {
                        name
                        rank
                        isAdult
                        category
                    }
                    meanScore
                }
            }
        }
        """

        ani_list = await self.request_anilist_batched(ani_query, mal_df["id"].to_list())

        return formatter.transform_ani_manga_data(ani_list, mal_df)

    async def request_anilist_batched(self, query, mal_ids):
        """Batch AniList requests to avoid query size limits.

        Uses request_single instead of request_paginated because AniList
        returns unreliable pagination data for idMal_in queries.
        Each batch of ANILIST_ID_BATCH_SIZE IDs fits in a single page.
        """
        all_media = []

        async with aiohttp.ClientSession() as session:
            for i in range(0, len(mal_ids), ANILIST_ID_BATCH_SIZE):
                batch = mal_ids[i : i + ANILIST_ID_BATCH_SIZE]
                result = await self.ani_provider.connection.request_single(
                    session, query, {"idMal_in": batch, "page": 1}
                )
                all_media.extend(result.get("data", {}).get("Page", {}).get("media", []))

        return {"data": {"media": all_media}}

    def get_nsfw_tags(self):
        return self.ani_provider.get_nsfw_tags()

    def get_tag_lookup(self):
        return self.ani_provider.get_tag_lookup()

    def get_genres(self):
        return self.ani_provider.get_genres()

    def get_related_anime(self, related_id):
        pass
