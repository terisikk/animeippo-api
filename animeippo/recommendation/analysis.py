import pandas as pd
import numpy as np
import scipy.spatial.distance as scdistance
import sklearn.preprocessing as skpre
import sklearn.cluster as skcluster


def unique_features_from_categoricals(*args):
    return pd.concat(args).explode().unique()


def distance(x_orig, y_orig, metric="jaccard"):
    return scdistance.cdist(x_orig, y_orig, metric=metric)


def similarity(x_orig, y_orig, metric="jaccard"):
    distances = distance(x_orig, y_orig, metric)
    return 1 - distances


def categorical_similarity(features1, features2, encoder, index=None):
    if index is None:
        index = features1.index

    return pd.DataFrame(
        similarity(encoder.encode(features1.values), encoder.encode(features2.values)),
        index=index,
    )


def similarity_of_anime_lists(features1, features2, encoder):
    similarities = categorical_similarity(features1, features2, encoder)

    return similarities.mean(axis=1, skipna=True)


def mean_score_per_categorical(dataframe, column):
    return dataframe.groupby(column)["score"].mean()


def weighted_mean_for_categorical_values(categoricals, weights, fillna=0.0):
    return np.nanmean([weights.get(categorical, fillna) for categorical in categoricals])


def weight_categoricals(dataframe, column):
    exploded = dataframe.explode(column)
    averages = mean_score_per_categorical(exploded, column)

    weights = np.sqrt(exploded[column].value_counts())
    weights = weights * averages

    return weights


def weight_categoricals_z_score(dataframe, column):
    df = dataframe.explode(column)

    scores = df.groupby(column)["score"].mean()
    counts = pd.DataFrame(df[column].value_counts())

    mean = np.mean(scores)
    std = np.std(scores)

    weighted_scores = counts.apply(lambda row: ((scores[row.name] - mean) / std) * row, axis=1)

    return normalize_column(weighted_scores)


def cluster_by_features(dataframe, column, encoder, model):
    model = skcluster.AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="average", distance_threshold=0.85
    )

    encoded = encoder.encode(dataframe[column])
    distances = pd.DataFrame(distance(encoded, encoded), index=dataframe.index)

    return model.fit_predict(distances), model.n_clusters_


def normalize_column(df_column):
    shaped = df_column.to_numpy().reshape(-1, 1)
    return pd.DataFrame(skpre.MinMaxScaler().fit_transform(shaped), index=df_column.index)
