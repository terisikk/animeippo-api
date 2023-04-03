import pandas as pd
import numpy as np
import sklearn.cluster as skcluster
import sklearn.metrics.pairwise as skpair
import scipy.spatial.distance as scdistance

import animeippo.providers.myanimelist as mal
import animeippo.recommendation.util as pdutil

NCLUSTERS = 10


def pairwise_distance(x, metric="jaccard"):
    encoded = pdutil.one_hot_categorical(x, mal.MAL_GENRES)

    return skpair.pairwise_distances(encoded, metric=metric)


def similarity(x_orig, y_orig, metric="jaccard"):
    encoded_x = pdutil.one_hot_categorical(x_orig, mal.MAL_GENRES)
    encoded_y = pdutil.one_hot_categorical(y_orig, mal.MAL_GENRES)

    distances = scdistance.cdist(encoded_x, encoded_y, metric=metric)

    return pd.DataFrame(1 - distances)


def similarity_of_anime_lists(dataframe1, dataframe2):
    similarities = similarity(dataframe1["genres"], dataframe2["genres"])
    similarities = similarities.apply(np.nanmean, axis=1)

    return similarities


def get_genre_clustering(dataframe, n_clusters=NCLUSTERS):
    model = skcluster.AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="average"
    )
    distance_matrix = pairwise_distance(dataframe["genres"])

    return model.fit(distance_matrix).labels_


def recommend_by_genre_similarity(target_df, source_df, weighted=False):
    similarities = similarity_of_anime_lists(target_df, source_df)

    if weighted:
        averages = genre_average_scores(source_df)
        similarities = pdutil.normalize_column(similarities) + (
            1.5
            * pdutil.normalize_column(
                target_df["genres"].apply(user_genre_weight, args=(averages,))
            )
        )
        similarities = similarities / 2

    target_df["recommend_score"] = similarities

    return target_df.sort_values("recommend_score", ascending=False)


def recommend_by_cluster(target_df, source_df, weighted=False):
    source_df["cluster"] = get_genre_clustering(source_df)

    scores = pd.DataFrame(index=target_df.index)

    for cluster_id, cluster in source_df.groupby("cluster"):
        similarities = similarity_of_anime_lists(target_df, cluster)

        if weighted:
            averages = cluster["user_score"].mean() / 10
            similarities = similarities * averages

        scores["cluster_" + str(cluster_id)] = similarities

    target_df["recommend_score"] = scores.apply(np.max, axis=1)

    return target_df.sort_values("recommend_score", ascending=False)


def genre_average_scores(dataframe):
    gdf = dataframe.explode("genres")

    return gdf.groupby("genres")["user_score"].mean()


def user_genre_weight(genres, averages):
    mean = np.nanmean([averages.get(genre, np.nan) for genre in genres])
    mean = mean if not np.isnan(mean) else 0.0

    return mean
