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


def jaccard_pairwise_distance(x):
    encoded = one_hot_genres(x)

    return skpair.pairwise_distances(encoded, metric="jaccard")


def jaccard_similarity(x_orig, y_orig):
    encoded_x = one_hot_genres(x_orig)
    encoded_y = one_hot_genres(y_orig)

    distances = scdistance.cdist(encoded_x, encoded_y, metric="jaccard")

    return pd.DataFrame(distances).applymap(lambda x: 1 - x)


def similarity_of_anime_lists(dataframe1, dataframe2):
    similarities = jaccard_similarity(dataframe1["genres"], dataframe2["genres"])
    similarities = similarities.apply(lambda row: row.mean(axis=0), axis=1)

    return similarities.sort_values(axis=0, ascending=False)


def create_genre_contingency_table(dataframe):
    gdf = dataframe.explode("genres")

    return pd.crosstab(gdf["genres"], gdf["cluster"])


def get_genre_clustering(dataframe, n_clusters=NCLUSTERS):
    model = skcluster.AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="average"
    )
    distance_matrix = jaccard_pairwise_distance(dataframe["genres"])

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


def order_by_recommendation(target_df, source_df):
    similarities = similarity_of_anime_lists(target_df, source_df)

    return target_df.reindex(index=similarities.index)
