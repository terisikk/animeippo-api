import pandas as pd


def data_from_mal(data):
    return pd.DataFrame(data)


def split_mal_genres(genres):
    genrenames = set()
    for genre in genres:
        genrenames.add(genre.get("name", None))

    return genrenames
