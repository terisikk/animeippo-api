import requests
import dotenv
import os

import animeippo.analysis as analysis

dotenv.load_dotenv("conf/prod.env")

MAL_API_URL = "https://api.myanimelist.net/v2"
MAL_API_TOKEN = os.environ.get("MAL_API_TOKEN", None)

HEADERS = {"Authorization": f"Bearer {MAL_API_TOKEN}"}
REQUEST_TIMEOUT = 30


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


def requests_get_all_pages(session, user):
    query_parameters = {
        "limit": 50,
        "nsfw": "true",
        "fields": "id,title,genres,",  # my_list_status{score,tags}",
    }

    response = session.get(
        f"{MAL_API_URL}/users/{user}/animelist",
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
        params=query_parameters,
    )

    response.raise_for_status()
    page = response.json()

    while page:
        yield page
        page = requests_get_next_page(session, page)


def get_anime_list(user):
    anime_list = []

    with requests.Session() as session:
        for page in requests_get_all_pages(session, user):
            for item in page["data"]:
                anime_list.append(item["node"])

    return anime_list


def analyze_mal(user):
    anime_list = get_anime_list(user)
    df, descriptions = analysis.transform_mal_data(anime_list)
    return df, descriptions
