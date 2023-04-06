import sklearn.preprocessing as skpre

import numpy as np

from . import filters


class AnimeRecommendationEngine:
    provider = None
    encoder = None
    scorer = None
    rec_filters = []

    def __init__(self, provider, scorer, encoder):
        self.provider = provider
        self.scorer = scorer
        self.encoder = encoder

    def recommend_seasonal_anime_for_user(self, user, year, season):
        seasonal_anime = self.provider.get_seasonal_anime_list(year, season)
        user_anime = self.provider.get_user_anime_list(user)

        user_anime_filtered = filters.IdFilter(*seasonal_anime["id"], negative=True).filter(
            user_anime
        )

        seasonal_anime_filtered = self.filter_anime(seasonal_anime)

        recommendations = self.scorer.score(
            seasonal_anime_filtered, user_anime_filtered, self.encoder
        )

        return recommendations

    def filter_anime(self, anime):
        filtered_df = anime

        for filter in self.rec_filters:
            filtered_df = filter.filter(filtered_df)

        return filtered_df

    def add_recommendation_filter(self, filter):
        self.rec_filters.append(filter)


class CategoricalEncoder:
    mlb = None

    def __init__(self, classes):
        self.mlb = skpre.MultiLabelBinarizer(classes=classes)
        self.mlb.fit(None)

    def encode(self, series):
        return np.array(self.mlb.transform(series), dtype=bool)
