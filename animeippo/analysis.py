import pandas as pd
import numpy as np
import sklearn.cluster as skcluster
import sklearn.metrics.pairwise as skpair
import sklearn.preprocessing as skpre
import scipy.spatial.distance as scdistance
import scipy.stats as scstats

import animeippo.providers.myanimelist as mal

NCLUSTERS = 10


def one_hot_genres(df_column):
    mlb = skpre.MultiLabelBinarizer(classes=mal.MAL_GENRES)
    mlb.fit(None)
    return np.array(mlb.transform(df_column), dtype=bool)


def pairwise_distance(x, metric="jaccard"):
    encoded = one_hot_genres(x)

    return skpair.pairwise_distances(encoded, metric=metric)


def similarity(x_orig, y_orig, metric="jaccard"):
    encoded_x = one_hot_genres(x_orig)
    encoded_y = one_hot_genres(y_orig)

    distances = scdistance.cdist(encoded_x, encoded_y, metric=metric)

    return pd.DataFrame(1 - distances)


def similarity_of_anime_lists(dataframe1, dataframe2):
    similarities = similarity(dataframe1["genres"], dataframe2["genres"])
    similarities = similarities.apply(lambda row: row.mean(axis=0), axis=1)

    return similarities


def create_genre_contingency_table(dataframe):
    gdf = dataframe.explode("genres")

    return pd.crosstab(gdf["genres"], gdf["cluster"])


def get_genre_clustering(dataframe, n_clusters=NCLUSTERS):
    model = skcluster.AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="average"
    )
    distance_matrix = pairwise_distance(dataframe["genres"])

    return model.fit(distance_matrix).labels_


def describe_clusters(dataframe, n_features):
    contingency_table = create_genre_contingency_table(dataframe)

    _, _, _, expected = scstats.chi2_contingency(contingency_table)

    squared_resid = calculate_residuals(contingency_table, expected)

    descriptions = contingency_table.apply(
        lambda row: squared_resid.nlargest(n_features, row.name).index.values, axis=0
    ).T

    return descriptions


def calculate_residuals(contingency_table, expected):
    residuals = (contingency_table - expected) / np.sqrt(expected)
    return residuals**2


def recommend_by_genre_similarity(target_df, source_df, weighted=False):
    similarities = similarity_of_anime_lists(target_df, source_df)

    if weighted:
        averages = genre_average_scores(source_df)
        similarities = normalize_column(similarities) + (
            1.5 * normalize_column(target_df["genres"].apply(user_genre_weight, args=(averages,)))
        )
        similarities = similarities / 2

    similarities = similarities.sort_values(0, ascending=False)
    target_df["recommend_score"] = similarities

    return target_df.reindex(index=similarities.index)


def recommend_by_cluster(target_df, source_df, weighted=False):
    source_df["cluster"] = get_genre_clustering(source_df)

    scores = pd.DataFrame(index=target_df.index)

    for cluster_id in source_df["cluster"].unique():
        cluster = source_df[source_df["cluster"] == cluster_id]

        similarities = similarity_of_anime_lists(target_df, cluster)

        if weighted:
            averages = cluster["user_score"].mean() / 10
            similarities = similarities * averages

        scores["cluster_" + str(cluster_id)] = similarities

    target_df["cluster"] = np.nan
    target_df["recommend_score"] = np.nan

    for i, row in scores.iterrows():
        m = row.max()
        target_df.at[i, "recommend_score"] = m

    return target_df.sort_values("recommend_score", ascending=False)


def genre_average_scores(dataframe):
    gdf = dataframe.explode("genres")

    return gdf.groupby("genres")["user_score"].mean()


def user_genre_weight(genres, averages):
    mean = np.nanmean([averages.get(genre, np.nan) for genre in genres])
    mean = mean if not np.isnan(mean) else 0.0

    return mean


def normalize_column(df_column):
    shaped = df_column.to_numpy().reshape(-1, 1)
    return pd.DataFrame(skpre.MinMaxScaler().fit_transform(shaped))
