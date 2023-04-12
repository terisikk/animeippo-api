import requests
import dotenv
import os
import pandas as pd
import numpy as np

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


class MyAnimeListProvider(provider.AbstractAnimeProvider):
    def get_user_anime_list(self, user_id):
        query = f"{MAL_API_URL}/users/{user_id}/animelist"
        parameters = {
            "limit": 50,
            "nsfw": "true",
            "fields": "id,title,genres,list_status{score,status},studios,rating{value},start_season",
        }

        anime_list = request_anime_list(query, parameters)

        return self.transform_to_animeippo_format(anime_list)

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
            "start_season",
            "rating{value}",
        ]
        parameters = {"limit": 50, "nsfw": "true", "fields": ",".join(fields)}

        anime_list = request_anime_list(query, parameters)

        return self.transform_to_animeippo_format(anime_list)

    def transform_to_animeippo_format(self, data):
        anime_list = []

        for item in data:
            anime = item["node"]

            list_status = item.get("list_status", np.nan)
            anime["list_status"] = list_status

            anime_list.append(anime)

        df = pd.DataFrame(anime_list)
        df["genres"] = df["genres"].apply(split_id_name_field)
        df["studios"] = df["studios"].apply(split_id_name_field)

        df["start_season"] = df["start_season"].apply(split_season)

        df["user_score"] = df["list_status"].apply(get_user_score)
        df["status"] = df["list_status"].apply(get_user_anime_status)
        df = df.drop("list_status", axis=1)

        df = df.drop(["main_picture"], axis=1)

        df = df.set_index("id")

        return df

    def get_genre_tags(self):
        return MAL_GENRES


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
    anime_list = []
    with requests.Session() as session:
        for page in requests_get_all_pages(session, query, parameters):
            for item in page["data"]:
                anime_list.append(item)

    return anime_list


def get_user_score(list_status):
    score = np.nan

    if list_status and isinstance(list_status, dict):
        score = list_status.get("score", np.nan)

        if score == 0:
            score = np.nan

    return score


def get_user_anime_status(list_status):
    status = np.nan

    if list_status and isinstance(list_status, dict):
        status = list_status.get("status", np.nan)

    return status


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
