import requests
import dotenv
import os
import pandas as pd

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
                anime = item["node"]
                anime["list_status"] = item.get("list_status", None)
                anime_list.append(anime)

    return anime_list


def get_user_anime(user):
    query = f"{MAL_API_URL}/users/{user}/animelist"
    parameters = {
        "limit": 50,
        "nsfw": "true",
        "fields": "id,title,genres,list_status{score}",
    }

    anime_list = request_anime_list(query, parameters)

    return transform_to_animeippo_format(anime_list)


def get_seasonal_anime(year=None, season=None):
    query = f"{MAL_API_URL}/anime/season/{year}/{season}"
    parameters = {
        "limit": 50,
        "nsfw": "true",
        "fields": "id,title,genres,media_type",
    }

    anime_list = request_anime_list(query, parameters)

    return transform_to_animeippo_format(anime_list)


def transform_to_animeippo_format(data):
    df = pd.DataFrame(data)
    df["genres"] = df["genres"].apply(split_mal_genres)

    df["user_score"] = df["list_status"].apply(get_user_score)

    df = df.drop("main_picture", axis=1)
    return df


def get_user_score(list_status):
    score = None

    if list_status:
        try:
            score = list_status.get("score", None)

            if score == 0:
                score = None
        except AttributeError as e:
            print(f"Could not extract genres from {list_status}: {e}")

    return score


def split_mal_genres(genres):
    genrenames = []

    if genres:
        try:
            for genre in genres:
                genrenames.append(genre.get("name", None))
        except TypeError as e:
            print(f"Could not extract genres from {genres}: {e}")

    return genrenames
