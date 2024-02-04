import datetime


def get_current_anime_season():
    today = datetime.date.today()

    season = ""

    if today.month in [1, 2, 3]:
        season = "winter"
    elif today.month in [4, 5, 6]:
        season = "spring"
    elif today.month in [7, 8, 9]:
        season = "summer"
    elif today.month in [10, 11, 12]:
        season = "fall"
    else:
        season = "?"

    return today.year, season
