import pandas as pd
import sklearn.cluster as skcluster


NCLUSTERS = 4


def transform_mal_data(data):
    df = pd.DataFrame(data)
    df["genres"] = df["genres"].apply(split_mal_genres)
    df["cluster"] = get_genre_clustering(df, NCLUSTERS)
    return df


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
    genrelist = get_genre_list(df_column)
    return pd.get_dummies(genrelist).groupby(level=0).sum()


def filter_by_cluster(dataframe, clusterno):
    return dataframe[dataframe.apply(lambda row: row["cluster"] == clusterno, axis=1)]


def describe_dataframe(dataframe):
    description = []
    g_column = dataframe["genres"]

    while len(g_column) > 0:
        genres = sort_genres_by_count(g_column)
        top_genre = genres.head(1).index[0]

        description.append(top_genre)
        g_column = g_column[g_column.apply(lambda g: top_genre not in g)]

    return description


def get_genre_clustering(dataframe, n_clusters=4):
    model = skcluster.AgglomerativeClustering(n_clusters=n_clusters)
    model.fit(one_hot_genres(dataframe["genres"]))

    return model.labels_
