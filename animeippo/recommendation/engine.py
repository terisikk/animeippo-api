import animeippo.providers as providers
import animeippo.providers.myanimelist as mal
import kmodes.kmodes as kmcluster
import kmodes.util.dissim as kdissim
import animeippo.recommendation.util as pdutil

from . import analysis


class ClusteringAnimeRecommendationEngine:
    NCLUSTERS = 10
    model = None
    provider = None

    def __init__(self, provider, clusters=NCLUSTERS):
        self.provider = provider
        self.model = kmcluster.KModes(n_clusters=clusters, cat_dissim=kdissim.ng_dissim, n_init=50)

    def recommend_seasonal_anime_for_user(self, user, year, season, weighted=True):
        seasonal_anime = self.provider.get_seasonal_anime_list(year, season)
        user_anime = self.provider.get_user_anime_list(user)

        filter_df = self.filter_seasonal_anime_from_user_anime(user_anime, seasonal_anime)

        self.model.fit_predict(
            pdutil.one_hot_categorical(filter_df["genres"], self.provider.get_genre_tags())
        )

        filter_df["cluster"] = self.model.labels_
        recommendations = analysis.score_by_cluster_similarity(seasonal_anime, filter_df, weighted)

        return recommendations[recommendations["media_type"] == "tv"]

    def filter_seasonal_anime_from_user_anime(self, user_anime, seasonal_anime):
        return user_anime[~user_anime["id"].isin(seasonal_anime["id"])]


class SimilarityAnimeRecommendationEngine:
    provider = None

    def __init__(self, provider):
        self.provider = provider

    def recommend_seasonal_anime_for_user(self, user, year, season, weighted=True):
        seasonal_anime = self.provider.get_seasonal_anime_list(year, season)
        user_anime = self.provider.get_user_anime_list(user)

        recommendations = analysis.score_by_genre_similarity(seasonal_anime, user_anime, weighted)

        return recommendations[recommendations["media_type"] == "tv"]
