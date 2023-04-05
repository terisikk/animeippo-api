import pandas as pd
import numpy as np
import scipy.spatial.distance as scdistance

import animeippo.providers.myanimelist as mal
import animeippo.recommendation.util as pdutil


def similarity(x_orig, y_orig, metric="jaccard"):
    distances = scdistance.cdist(x_orig, y_orig, metric=metric)

    return pd.DataFrame(1 - distances)


def similarity_of_anime_lists(dataframe1, dataframe2, encoder):
    similarities = similarity(
        encoder.encode(dataframe1["genres"]), encoder.encode(dataframe2["genres"])
    )
    similarities = similarities.apply(np.nanmean, axis=1)

    return similarities


def genre_average_scores(dataframe):
    gdf = dataframe.explode("genres")

    return gdf.groupby("genres")["user_score"].mean()


def user_genre_weight(genres, averages):
    mean = np.nanmean([averages.get(genre, np.nan) for genre in genres])
    mean = mean if not np.isnan(mean) else 0.0

    return mean
