import kmodes.kmodes as kmcluster
import kmodes.util.dissim as kdissim
import animeippo.recommendation.util as pdutil
import sklearn.preprocessing as skpre
import scipy.spatial.distance as scdistance

import numpy as np

from . import analysis


class ClusteringAnimeRecommendationEngine:
    NCLUSTERS = 10
    model = None
    provider = None
    encoder = None

    def __init__(self, provider, clusters=NCLUSTERS):
        self.provider = provider

        self.encoder = CategoricalEncoder(classes=self.provider.get_genre_tags())

        self.model = kmcluster.KModes(n_clusters=clusters, cat_dissim=kdissim.ng_dissim, n_init=50)

    def recommend_seasonal_anime_for_user(self, user, year, season, weighted=True):
        seasonal_anime = self.provider.get_seasonal_anime_list(year, season)
        user_anime = self.provider.get_user_anime_list(user)

        filter_df = self.filter_seasonal_anime_from_user_anime(user_anime, seasonal_anime)

        self.model.fit_predict(self.encoder.encode(filter_df["genres"]))

        filter_df["cluster"] = self.model.labels_

        recommendations = analysis.score_by_cluster_similarity(
            self.encoder, seasonal_anime, filter_df, weighted
        )

        return recommendations[recommendations["media_type"] == "tv"]

    def filter_seasonal_anime_from_user_anime(self, user_anime, seasonal_anime):
        return user_anime[~user_anime["id"].isin(seasonal_anime["id"])]


class SimilarityAnimeRecommendationEngine:
    provider = None
    encoder = None

    def __init__(self, provider):
        self.provider = provider

        self.encoder = CategoricalEncoder(classes=self.provider.get_genre_tags())

    def recommend_seasonal_anime_for_user(self, user, year, season, weighted=True):
        seasonal_anime = self.provider.get_seasonal_anime_list(year, season)
        user_anime = self.provider.get_user_anime_list(user)

        recommendations = analysis.score_by_genre_similarity(
            self.encoder, seasonal_anime, user_anime, weighted
        )

        return recommendations[recommendations["media_type"] == "tv"]


class CategoricalEncoder:
    mlb = None

    def __init__(self, classes):
        self.mlb = skpre.MultiLabelBinarizer(classes=classes)
        self.mlb.fit(None)

    def encode(self, series):
        return np.array(self.mlb.transform(series), dtype=bool)
