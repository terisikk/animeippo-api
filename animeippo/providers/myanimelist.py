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


def get_user_anime(user):
    anime_list = []

    query = f"{MAL_API_URL}/users/{user}/animelist"
    query_parameters = {
        "limit": 50,
        "nsfw": "true",
        "fields": "id,title,genres,",  # my_list_status{score,tags}",
    }

    with requests.Session() as session:
        for page in requests_get_all_pages(session, query, query_parameters):
            for item in page["data"]:
                anime_list.append(item["node"])

    return transform_to_animeippo_format(anime_list)


def get_seasonal_anime(year=None, season=None):
    anime_list = []

    query = f"{MAL_API_URL}/anime/season/{year}/{season}"
    query_parameters = {
        "limit": 50,
        "nsfw": "true",
        "fields": "id,title,genres,media_type",  # my_list_status{score,tags}",
    }

    with requests.Session() as session:
        for page in requests_get_all_pages(session, query, query_parameters):
            for item in page["data"]:
                anime_list.append(item["node"])

    return transform_to_animeippo_format(anime_list)


def transform_to_animeippo_format(data):
    df = pd.DataFrame(data)
    df["genres"] = df["genres"].apply(split_mal_genres)
    df = df.drop("main_picture", axis=1)
    return df


def split_mal_genres(genres):
    genrenames = []

    try:
        for genre in genres:
            genrenames.append(genre.get("name", None))
    except TypeError as e:
        print(e)
        print(f"Could not extract genres from {genres}")

    return genrenames


def reduce_genres_to_mal_genres(genres):
    return [genre.capitalize() for genre in genres if genre.capitalize() in MAL_GENRES]
