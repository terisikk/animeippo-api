import pandas as pd
import sklearn.cluster as skcluster
import sklearn.metrics.pairwise as skpair
import sklearn.preprocessing as skpre
import scipy.spatial.distance as scdistance

import animeippo.providers.myanimelist as mal

NCLUSTERS = 10


def one_hot_genres(df_column, classes=mal.MAL_GENRES):
    mlb = skpre.MultiLabelBinarizer(classes=classes)
    return mlb.fit_transform(df_column)


def jaccard_pairwise_distance(x):
    encoded = one_hot_genres(x)

    return skpair.pairwise_distances(encoded, metric="jaccard")


def jaccard_spatial_distance(x_orig, y_orig, classes=mal.MAL_GENRES):
    mlb = skpre.MultiLabelBinarizer(classes=classes)
    encoded_x = mlb.fit_transform(x_orig)
    encoded_y = mlb.fit_transform(y_orig)

    distances = []

    for x in encoded_x:
        q_distance = []
        for y in encoded_y:
            dist = scdistance.jaccard(x, y)
            q_distance.append(dist)
        distances.append(q_distance)

    return pd.DataFrame(distances)


def jaccard_similarity_for_anime_lists(dataframe1, dataframe2):
    distances = jaccard_spatial_distance(dataframe1["genres"], dataframe2["genres"])
    similarities = distances.applymap(lambda x: 1 - x)

    similarities = similarities.set_axis(dataframe1["title"], copy=False, axis="index")
    similarities = similarities.set_axis(dataframe2["title"], copy=False, axis="columns")

    return similarities


def create_contingency_table(dataframe, x, y):
    all_categories = dataframe[x].apply(pd.Series).stack().unique()
    unique_y = dataframe[y].unique()

    contingency_table = pd.DataFrame(index=list(all_categories), columns=unique_y)
    for column in unique_y:
        column_category = dataframe[dataframe[y] == column]
        for category in all_categories:
            count = sum([category in categories for categories in column_category[x]])
            contingency_table.loc[category, column] = count

    return contingency_table


def get_genre_clustering(dataframe, n_clusters=NCLUSTERS):
    model = skcluster.AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="average"
    )
    distance_matrix = jaccard_pairwise_distance(dataframe["genres"])
    model.fit(distance_matrix)

    return model.labels_


def describe_clusters(dataframe, n_features):
    clusters = dataframe["cluster"].unique()
    descriptions = pd.DataFrame(index=clusters, columns=["description"])

    contingency_table = create_contingency_table(dataframe, "genres", "cluster")

    for cluster_id in clusters:
        deviations = find_genre_deviations(contingency_table, cluster_id)
        description = deviations.nlargest(n_features, "deviation").index.values
        descriptions.at[cluster_id, "description"] = description

    return descriptions.sort_index()


def find_genre_deviations(contingency, cluster_id):
    deviations = pd.DataFrame(index=contingency.index, columns=["deviation"], dtype="float64")

    sum_total = contingency.to_numpy().sum()
    sum_cluster = contingency[cluster_id].sum()

    for index, row in contingency.iterrows():
        deviation = calculate_deviation(sum_total, sum_cluster, row.sum(), row[cluster_id])
        deviations.at[index, "deviation"] = deviation

    return deviations


def calculate_deviation(total_sum, cluster_sum, row_sum, cell_value):
    return cell_value - ((row_sum * cluster_sum) / total_sum)


def order_by_recommendation(target_df, source_df):
    simis = jaccard_similarity_for_anime_lists(target_df, source_df)

    avgsim = pd.DataFrame(data=[row.mean(axis=0) for _, row in simis.iterrows()])
    target_df["avgsim"] = avgsim[0]

    return target_df.sort_values("avgsim", ascending=False).drop("avgsim", axis=1)
