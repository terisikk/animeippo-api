from datetime import timedelta

from . import provider
from . import myanimelist as mal
from . import anilist as ani
from .formatters import mixed_formatter


import animeippo.cache as animecache


class MixedProvider(provider.AbstractAnimeProvider):
    def __init__(self, cache=None):
        self.cache = cache

        self.ani_provider = ani.AniListProvider(cache)
        self.mal_provider = mal.MyAnimeListProvider(cache)

    @animecache.cached_dataframe(ttl=timedelta(days=1))
    async def get_user_anime_list(self, user_id):
        if not user_id:
            return None

        mal_query = f"/users/{user_id}/animelist"
        fields = [
            "id",
            "list_status{score,status,finish_date}",
        ]

        parameters = {"nsfw": "true", "fields": ",".join(fields)}

        mal_list = await self.mal_provider.connection.request_anime_list(mal_query, parameters)
        mal_df = mixed_formatter.transform_mal_watchlist_data(mal_list, [])

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
                    }
                    meanScore
                    source
                    studios { edges { node { name isAnimationStudio } }}
                    seasonYear
                    season
                    coverImage { large }
                }
            }
        }
        """

        variables = {"idMal_in": mal_df["id"].to_list()}

        ani_list = await self.ani_provider.connection.request_paginated(ani_query, variables)

        return mixed_formatter.transform_ani_watchlist_data(
            ani_list, self.get_feature_fields(), mal_df
        )

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

            variables = {"seasonYear": int(year), "season": str(season).upper()}

        anime_list = await self.ani_provider.connection.request_paginated(query, variables)

        return mixed_formatter.transform_ani_seasonal_data(anime_list, self.get_feature_fields())

    async def get_user_manga_list(self, user_id):
        return await self.mal_provider.get_user_manga_list(user_id)

    def get_feature_fields(self):
        return self.ani_provider.get_feature_fields()

    def get_related_anime(self, id):
        pass
