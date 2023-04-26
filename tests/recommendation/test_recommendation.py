import animeippo.recommendation.engine as engine
import animeippo.recommendation.scoring as scoring
import animeippo.providers.myanimelist as mal
from animeippo.recommendation import dataset
import pandas as pd
import pytest

from tests import test_data


class ProviderStub:
    def get_seasonal_anime_list(self, *args, **kwargs):
        return pd.DataFrame(test_data.FORMATTED_MAL_SEASONAL_LIST).set_index("id")

    def get_user_anime_list(self, *args, **kwargs):
        return pd.DataFrame(test_data.FORMATTED_MAL_USER_LIST).set_index("id")

    def get_related_anime(self, *args, **kwargs):
        return pd.DataFrame()


def test_recommend_seasonal_anime_for_user_by_genre():
    scorer = scoring.GenreSimilarityScorer(mal.MAL_GENRES)
    recengine = engine.AnimeRecommendationEngine()

    recengine.add_scorer(scorer)

    provider = ProviderStub()
    data = dataset.UserDataSet(provider.get_user_anime_list(), provider.get_seasonal_anime_list())

    recommendations = recengine.fit_predict(data)

    assert recommendations["title"].tolist() == [
        "Shingeki no Kyojin: The Final Season",
        "Golden Kamuy 4th Season",
    ]


def test_recommend_seasonal_anime_for_user_by_cluster():
    scorer = scoring.ClusterSimilarityScorer(mal.MAL_GENRES)

    recengine = engine.AnimeRecommendationEngine()
    recengine.add_scorer(scorer)

    provider = ProviderStub()
    data = dataset.UserDataSet(provider.get_user_anime_list(), provider.get_seasonal_anime_list())

    recommendations = recengine.fit_predict(data)

    assert recommendations["title"].tolist() == [
        "Golden Kamuy 4th Season",
        "Shingeki no Kyojin: The Final Season",
    ]


def test_multiple_scorers_can_be_added():
    scorer = scoring.GenreSimilarityScorer(mal.MAL_GENRES)
    scorer2 = scoring.StudioCountScorer()
    recengine = engine.AnimeRecommendationEngine()

    recengine.add_scorer(scorer)
    recengine.add_scorer(scorer2)

    provider = ProviderStub()
    data = dataset.UserDataSet(provider.get_user_anime_list(), provider.get_seasonal_anime_list())

    recommendations = recengine.fit_predict(data)

    assert recommendations["title"].tolist() == [
        "Shingeki no Kyojin: The Final Season",
        "Golden Kamuy 4th Season",
    ]


def test_runtime_error_is_raised_when_no_scorers_exist():
    recengine = engine.AnimeRecommendationEngine()

    data = dataset.UserDataSet(None, None)

    with pytest.raises(RuntimeError):
        recengine.fit_predict(data)
