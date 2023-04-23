import requests
import dotenv
import os
import pandas as pd
import numpy as np
import contextlib

from functools import lru_cache

from . import provider

dotenv.load_dotenv("conf/prod.env")

MAL_API_URL = "https://api.myanimelist.net/v2"
MAL_API_TOKEN = os.environ.get("MAL_API_TOKEN", None)

HEADERS = {"Authorization": f"Bearer {MAL_API_TOKEN}"}
REQUEST_TIMEOUT = 30

MAL_GENRES = [
    "Action",
    "Adult Cast",
    "Adventure",
    "Anthropomorphic",
    "Avant Garde",
    "Award Winning",
    "Boys Love",
    "CGDCT",
    "Childcare",
    "Combat Sports",
    "Comedy",
    "Crossdressing",
    "Delinquents",
    "Detective",
    "Drama",
    "Educational",
    "Erotica",
    "Fantasy",
    "Gag Humor",
    "Girls Love",
    "Gore",
    "Gourmet",
    "Harem",
    "Hentai",
    "High Stakes Game",
    "Historical",
    "Horror",
    "Idols (Female)",
    "Idols (Male)",
    "Isekai",
    "Iyashikei",
    "Josei",
    "Kids",
    "Love Polygon",
    "Magical Sex Shift",
    "Mahou Shoujo",
    "Martial Arts",
    "Mecha",
    "Medical",
    "Military",
    "Music",
    "Mystery",
    "Mythology",
    "Organized Crime",
    "Otaku Culture",
    "Parody",
    "Performing Arts",
    "Pets",
    "Psychological",
    "Racing",
    "Reincarnation",
    "Reverse Harem",
    "Romance",
    "Romantic Subtext",
    "Samurai",
    "School",
    "Sci-Fi",
    "Seinen",
    "Shoujo",
    "Shounen",
    "Showbiz",
    "Slice of Life",
    "Space",
    "Sports",
    "Strategy Game",
    "Super Power",
    "Supernatural",
    "Survival",
    "Suspense",
    "Ecchi",
    "Team Sports",
    "Time Travel",
    "Vampire",
    "Video Game",
    "Visual Arts",
    "Workplace",
]


@contextlib.contextmanager
def mal_session():
    with requests.Session() as session:
        yield session


class MyAnimeListProvider(provider.AbstractAnimeProvider):
    @lru_cache(maxsize=10)
    def get_user_anime_list(self, user_id):
        query = f"{MAL_API_URL}/users/{user_id}/animelist"
        fields = [
            "id",
            "title",
            "genres",
            "list_status{score,status}",
            "studios",
            "rating{value}",
            "start_season",
        ]

        parameters = {"limit": 50, "nsfw": "true", "fields": ",".join(fields)}

        anime_list = request_anime_list(query, parameters)

        return self.transform_to_animeippo_format(anime_list)

    @lru_cache(maxsize=10)
    def get_seasonal_anime_list(self, year, season):
        query = f"{MAL_API_URL}/anime/season/{year}/{season}"
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
        ]
        parameters = {"limit": 50, "nsfw": "true", "fields": ",".join(fields)}

        anime_list = request_anime_list(query, parameters)

        return self.transform_to_animeippo_format(anime_list)

    def get_genre_tags(self):
        return MAL_GENRES

    @lru_cache(maxsize=128)
    def get_related_anime(self, anime_id):
        query = f"{MAL_API_URL}/anime/{anime_id}"

        fields = [
            "id",
            "title",
            "source",
            "related_anime",
        ]
        parameters = {"fields": ",".join(fields)}

        anime_list = request_related_anime(query, parameters)

        meaningful_relations = ["parent_story", "prequel"]

        related_anime = pd.DataFrame()

        if len(anime_list["data"]) > 0:
            related_anime = self.transform_to_animeippo_format(anime_list)

            related_anime = related_anime[related_anime["relation_type"].isin(meaningful_relations)]

        return related_anime.index.tolist()

    def transform_to_animeippo_format(self, data):
        df = pd.json_normalize(data["data"], max_level=1)

        for key, formatter in formatters.items():
            if key in df.columns:
                df[key] = df[key].apply(formatter)

        column_mapping = {}

        for key in df.columns:
            if "." in key:
                new_key = key.split(".")[-1]
                column_mapping[key] = new_key

        df = df.rename(columns=column_mapping)

        dropped = [
            "num_episodes_watched",
            "is_rewatching",
            "updated_at",
            "start_date",
            "finish_date",
        ]

        df = df.drop(dropped, errors="ignore", axis=1)

        df["related_anime"] = np.empty((len(df.index), 0)).tolist()

        if "id" in df.columns:
            df = df.set_index("id")

        return df


def requests_get_next_page(session, page):
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


def requests_get_all_pages(session, query, parameters):
    response = session.get(
        query,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
        params=parameters,
    )

    response.raise_for_status()
    page = response.json()

    while page:
        yield page
        page = requests_get_next_page(session, page)


def request_anime_list(query, parameters):
    anime_list = {"data": []}
    with mal_session() as session:
        for page in requests_get_all_pages(session, query, parameters):
            for item in page["data"]:
                anime_list["data"].append(item)

        return anime_list


def request_related_anime(query, parameters):
    anime = {"data": []}

    with mal_session() as session:
        response = session.get(
            query,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
            params=parameters,
        )

        response.raise_for_status()
        details = response.json()
        anime["data"] = details["related_anime"]

    return anime


def get_user_score(score):
    if score == 0:
        score = np.nan

    return score


def split_id_name_field(field):
    names = []

    if field:
        try:
            for item in field:
                if isinstance(item, dict):
                    names.append(item.get("name", np.nan))
        except TypeError as e:
            print(f"Could not extract items from {field}: {e}")

    return names


def split_season(season_field):
    season_ret = np.nan

    if season_field and isinstance(season_field, dict):
        year = season_field.get("year", "?")
        season = season_field.get("season", "?")

        season_ret = f"{year}/{season}"

    return season_ret


formatters = {
    "node.genres": split_id_name_field,
    "node.studios": split_id_name_field,
    "node.start_season": split_season,
    "list_status.score": get_user_score,
}
