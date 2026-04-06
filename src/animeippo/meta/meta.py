import datetime


def get_current_anime_season():
    today = datetime.date.today()

    season = ""

    if today.month in [1, 2, 3]:
        season = "WINTER"
    elif today.month in [4, 5, 6]:
        season = "SPRING"
    elif today.month in [7, 8, 9]:
        season = "SUMMER"
    elif today.month in [10, 11, 12]:
        season = "FALL"
    else:
        season = "?"

    return today.year, season
