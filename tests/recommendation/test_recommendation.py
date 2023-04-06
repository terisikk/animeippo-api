import animeippo.recommendation.engine as engine
import animeippo.recommendation.scoring as scoring
import animeippo.providers.myanimelist as mal
import animeippo.recommendation.filters as filters
import pandas as pd

from tests import test_data


# Figure out how to provide correct data from this, mal data is not formatted
class ProviderStub:
    def get_seasonal_anime_list(self, *args, **kwargs):
        return pd.DataFrame(test_data.FORMATTED_MAL_SEASONAL_LIST)

    def get_user_anime_list(self, *args, **kwargs):
        return pd.DataFrame(test_data.FORMATTED_MAL_USER_LIST)


def test_recommend_seasonal_anime_for_user_by_genre(requests_mock):
    user = "Janiskeisari"
    year = "2023"
    season = "winter"

    encoder = engine.CategoricalEncoder(mal.MAL_GENRES)
    scorer = scoring.GenreSimilarityScorer()
    recengine = engine.AnimeRecommendationEngine(ProviderStub(), scorer, encoder)

    recommendations = recengine.recommend_seasonal_anime_for_user(user, year, season)

    assert recommendations["title"].tolist() == [
        "Shingeki no Kyojin: The Final Season",
        "Golden Kamuy 4th Season",
    ]


def test_recommend_seasonal_anime_for_user_by_cluster():
    user = "Janiskeisari"
    year = "2023"
    season = "winter"

    encoder = engine.CategoricalEncoder(mal.MAL_GENRES)
    scorer = scoring.ClusterSimilarityScorer(2)
    recengine = engine.AnimeRecommendationEngine(ProviderStub(), scorer, encoder)
    recommendations = recengine.recommend_seasonal_anime_for_user(user, year, season)

    assert recommendations["title"].tolist() == [
        "Golden Kamuy 4th Season",
        "Shingeki no Kyojin: The Final Season",
    ]


def test_filters_work():
    user = "Janiskeisari"
    year = "2023"
    season = "winter"

    encoder = engine.CategoricalEncoder(mal.MAL_GENRES)
    scorer = scoring.ClusterSimilarityScorer(2)
    recengine = engine.AnimeRecommendationEngine(ProviderStub(), scorer, encoder)
    recengine.add_recommendation_filter(filters.GenreFilter("Gore", negative=True))

    recommendations = recengine.recommend_seasonal_anime_for_user(user, year, season)

    assert len(recommendations.index) == 1
