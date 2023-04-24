import animeippo.recommendation.engine as engine
import animeippo.recommendation.scoring as scoring
import animeippo.providers.myanimelist as mal
import animeippo.recommendation.filters as filters
import pandas as pd
import pytest

from tests import test_data


# Figure out how to provide correct data from this, mal data is not formatted
class ProviderStub:
    def get_seasonal_anime_list(self, *args, **kwargs):
        return pd.DataFrame(test_data.FORMATTED_MAL_SEASONAL_LIST).set_index("id")

    def get_user_anime_list(self, *args, **kwargs):
        return pd.DataFrame(test_data.FORMATTED_MAL_USER_LIST).set_index("id")

    def get_related_anime(self, *args, **kwargs):
        return []


def test_recommend_seasonal_anime_for_user_by_genre():
    user = "Janiskeisari"
    year = "2023"
    season = "winter"

    encoder = scoring.CategoricalEncoder(mal.MAL_GENRES)
    scorer = scoring.GenreSimilarityScorer(encoder)
    recengine = engine.AnimeRecommendationEngine(ProviderStub())

    recengine.add_scorer(scorer)

    recommendations = recengine.recommend_seasonal_anime_for_user(user, year, season)

    assert recommendations["title"].tolist() == [
        "Shingeki no Kyojin: The Final Season",
        "Golden Kamuy 4th Season",
    ]


def test_recommend_seasonal_anime_for_user_by_cluster():
    user = "Janiskeisari"
    year = "2023"
    season = "winter"

    encoder = scoring.CategoricalEncoder(mal.MAL_GENRES)
    scorer = scoring.ClusterSimilarityScorer(encoder)

    recengine = engine.AnimeRecommendationEngine(ProviderStub())
    recengine.add_scorer(scorer)

    recommendations = recengine.recommend_seasonal_anime_for_user(user, year, season)

    assert recommendations["title"].tolist() == [
        "Golden Kamuy 4th Season",
        "Shingeki no Kyojin: The Final Season",
    ]


def test_filters_work():
    user = "Janiskeisari"
    year = "2023"
    season = "winter"

    encoder = scoring.CategoricalEncoder(mal.MAL_GENRES)
    scorer = scoring.ClusterSimilarityScorer(encoder)
    recengine = engine.AnimeRecommendationEngine(ProviderStub())

    recengine.add_scorer(scorer)
    recengine.add_recommendation_filter(filters.GenreFilter("Gore", negative=True))

    recommendations = recengine.recommend_seasonal_anime_for_user(user, year, season)

    assert len(recommendations.index) == 1


def test_multiple_scorers_can_be_added():
    user = "Janiskeisari"
    year = "2023"
    season = "winter"

    encoder = scoring.CategoricalEncoder(mal.MAL_GENRES)
    scorer = scoring.GenreSimilarityScorer(encoder)
    scorer2 = scoring.StudioCountScorer()
    recengine = engine.AnimeRecommendationEngine(ProviderStub())

    recengine.add_scorer(scorer)
    recengine.add_scorer(scorer2)

    recommendations = recengine.recommend_seasonal_anime_for_user(user, year, season)

    assert recommendations["title"].tolist() == [
        "Shingeki no Kyojin: The Final Season",
        "Golden Kamuy 4th Season",
    ]


def test_runtime_error_is_raised_when_no_scorers_exist():
    user = "Janiskeisari"
    year = "2023"
    season = "winter"

    recengine = engine.AnimeRecommendationEngine(ProviderStub())

    with pytest.raises(RuntimeError):
        recengine.recommend_seasonal_anime_for_user(user, year, season)
