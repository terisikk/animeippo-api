import requests
import dotenv
import os
import contextlib

from datetime import timedelta

from . import provider
from .formatters import mal_formatter

import animeippo.cache as animecache

dotenv.load_dotenv("conf/prod.env")

MAL_API_URL = "https://api.myanimelist.net/v2"
MAL_API_TOKEN = os.environ.get("MAL_API_TOKEN", None)

HEADERS = {"Authorization": f"Bearer {MAL_API_TOKEN}"}
REQUEST_TIMEOUT = 30


@contextlib.contextmanager
def mal_session():
    with requests.Session() as session:
        yield session


class MyAnimeListProvider(provider.AbstractAnimeProvider):
    def __init__(self, cache=None):
        self.connection = MyAnimeListConnection(cache)
        self.cache = cache

    @animecache.cached_dataframe(ttl=timedelta(days=1))
    def get_user_anime_list(self, user_id):
        query = f"/users/{user_id}/animelist"
        fields = [
            "id",
            "title",
            "genres",
            "list_status{score,status}",
            "studios",
            "rating{value}",
            "start_season",
            "source",
        ]

        parameters = {"nsfw": "true", "fields": ",".join(fields)}

        anime_list = self.connection.request_anime_list(query, parameters)

        return mal_formatter.transform_to_animeippo_format(anime_list)

    @animecache.cached_dataframe(ttl=timedelta(days=1))
    def get_seasonal_anime_list(self, year, season):
        query = f"/anime/season/{year}/{season}"
        fields = [
            "id",
            "title",
            "genres",
            "media_type",
            "studios",
            "mean",
            "num_list_users",
            "rating{value}",
            "start_season",
            "source",
        ]
        parameters = {"nsfw": "true", "fields": ",".join(fields)}

        anime_list = self.connection.request_anime_list(query, parameters)

        return mal_formatter.transform_to_animeippo_format(anime_list)

    def get_related_anime(self, anime_id):
        query = f"/anime/{anime_id}"

        fields = [
            "id",
            "title",
            "source",
            "related_anime",
        ]
        parameters = {"fields": ",".join(fields)}

        anime_list = self.connection.request_related_anime(query, parameters)

        return mal_formatter.transform_to_animeippo_format(anime_list)

    def get_features(self):
        return ["genres", "rating"]


class MyAnimeListConnection:
    def __init__(self, cache=None):
        self.cache = cache

    @animecache.cached_query(ttl=timedelta(days=1))
    def request_anime_list(self, query, parameters):
        anime_list = {"data": []}

        with mal_session() as session:
            for page in self.requests_get_all_pages(session, query, parameters):
                for item in page["data"]:
                    anime_list["data"].append(item)

        return anime_list

    @animecache.cached_query(ttl=timedelta(days=7))
    def request_related_anime(self, query, parameters):
        anime = {"data": []}

        with mal_session() as session:
            response = session.get(
                MAL_API_URL + query,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
                params=parameters,
            )

            response.raise_for_status()
            details = response.json()
            anime["data"] = details["related_anime"]

        return anime

    def requests_get_next_page(self, session, page):
        if page:
            next_page = None
            next_page_url = page.get("paging", dict()).get("next", None)

            if next_page_url:
                response = session.get(
                    next_page_url,
                    headers=HEADERS,
                    timeout=REQUEST_TIMEOUT,
                )

                response.raise_for_status()
                next_page = response.json()
                return next_page

    def requests_get_all_pages(self, session, query, parameters):
        response = session.get(
            MAL_API_URL + query,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
            params=parameters,
        )

        response.raise_for_status()
        page = response.json()

        while page:
            yield page
            page = self.requests_get_next_page(session, page)
