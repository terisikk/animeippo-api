import pandas as pd
import sklearn.cluster as skcluster
import sklearn.metrics.pairwise as skpair
import sklearn.preprocessing as skpre
import scipy.spatial.distance as scdistance
import scipy.stats as scstats


NCLUSTERS = 10


def transform_mal_data(data):
    df = pd.DataFrame(data)
    df["genres"] = df["genres"].apply(split_mal_genres)
    df = df.drop("main_picture", axis=1)
    df["cluster"] = get_genre_clustering(df, NCLUSTERS)
    descriptions = describe_clusters(df, 2)
    return df, descriptions


def split_mal_genres(genres):
    genrenames = []
    for genre in genres:
        genrenames.append(genre.get("name", None))

    return genrenames


def get_genre_list(df_column):
    genrelist = df_column.apply(pd.Series).stack()
    return genrelist.reset_index(level=1, drop=True).to_frame("genre")


def sort_genres_by_count(df_column):
    return df_column.apply(pd.Series).stack().value_counts()


def one_hot_genres(df_column):
    mlb = skpre.MultiLabelBinarizer()
    return mlb.fit_transform(df_column)


def jaccard_distance(df_column):
    encoded = one_hot_genres(df_column)
    return skpair.pairwise_distances(encoded, metric="jaccard")


def filter_by_cluster(dataframe, clusterno):
    return dataframe[dataframe["cluster"] == clusterno]


def describe_dataframe(dataframe):
    description = []
    g_column = dataframe["genres"]

    while len(g_column) > 0:
        genres = sort_genres_by_count(g_column)
        top_genre = genres.head(1).index[0]

        description.append(top_genre)
        g_column = g_column[g_column.apply(lambda g: top_genre not in g)]

    return description


def describe_clusters(dataframe, n_features):
    clusters = dataframe["cluster"].unique()
    descriptions = pd.DataFrame(index=clusters, columns=["description"])

    contingency_table = create_contingency_table(dataframe, "genres", "cluster")

    for cluster_id in clusters:
        deviations = find_genre_deviations(contingency_table, cluster_id)
        description = deviations.nlargest(n_features, "deviation").index.values
        descriptions.at[cluster_id, "description"] = description

    return descriptions


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


def get_genre_clustering(dataframe, n_clusters=NCLUSTERS):
    model = skcluster.AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="average"
    )
    distance_matrix = jaccard_distance(dataframe["genres"])
    model.fit(distance_matrix)

    print(model.n_features_in_)

    return model.labels_
