import requests

from datetime import timedelta

from . import provider
from .formatters import ani_formatter

import animeippo.cache as animecache

REQUEST_TIMEOUT = 30


class AniListProvider(provider.AbstractAnimeProvider):
    def __init__(self, cache=None):
        self.connection = AnilistConnection(cache)

    def get_user_anime_list(self, user_id):
        # Here we define our query as a multi-line string
        query = """
        query ($userName: String) {
            MediaListCollection(userName: $userName, type: ANIME) {
                lists {
                    name
                    status
                    entries {
                        status
                        score(format:POINT_10)
                        media {
                            id
                            title { romaji }
                            genres
                            meanScore
                            source
                            studios { edges { id } }
                            seasonYear
                            season
                            coverImage { large }
                        }
                    }
                }
            }
        }
        """

        variables = {"userName": user_id}

        anime_list = self.connection.request_collection(query, variables)
        return ani_formatter.transform_to_animeippo_format(anime_list, normalize_level=1)

    def get_seasonal_anime_list(self, year, season):
        # Here we define our query as a multi-line string
        query = """
        query ($seasonYear: Int, $season: MediaSeason, $page: Int) {
            Page(page: $page, perPage: 50) {
                pageInfo { hasNextPage currentPage lastPage total perPage }
                media(seasonYear: $seasonYear, season: $season, type:ANIME) {
                    id
                    title { romaji }
                    genres
                    meanScore
                    source
                    studios { edges { id }}
                    seasonYear
                    season
                    relations { edges { relationType, node { id }}}
                    popularity
                    coverImage { large }
                }
            }
        }
        """

        variables = {"seasonYear": int(year), "season": str(season).upper()}

        anime_list = self.connection.request_paginated(query, variables)

        return ani_formatter.transform_to_animeippo_format(anime_list, normalize_level=0)

    def get_features(self):
        return ["genres"]

    def get_related_anime(self, id):
        pass


class AnilistConnection:
    def __init__(self, cache=None):
        self.cache = cache

    @animecache.cached_query(ttl=timedelta(days=1))
    def request_paginated(self, query, parameters):
        anime_list = {"data": []}
        variables = parameters.copy()  # To avoid cache miss with side effects

        for page in self.requests_get_all_pages(query, variables):
            for item in page["media"]:
                anime_list["data"].append(item)

        return anime_list

    @animecache.cached_query(ttl=timedelta(days=1))
    def request_collection(self, query, parameters):
        anime_list = {"data": []}
        variables = parameters.copy()  # To avoid cache miss with side effects

        for coll in self.request_single(query, variables)["data"]["MediaListCollection"]["lists"]:
            for entry in coll["entries"]:
                anime_list["data"].append(entry)

        return anime_list

    def request_single(self, query, variables):
        url = "https://graphql.anilist.co"

        response = requests.post(
            url, json={"query": query, "variables": variables}, timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()
        return response.json()

    def requests_get_all_pages(self, query, variables):
        variables["page"] = 0
        variables["perPage"] = 50

        page = self.request_single(query, variables)["data"]["Page"]

        safeguard = 10

        yield page

        while page["pageInfo"].get("hasNextPage", False) and safeguard > 0:
            variables["page"] = page["pageInfo"]["currentPage"] + 1

            page = self.request_single(query, variables)["data"]["Page"]
            yield page
            safeguard = safeguard - 1
