import animeippo.recommendation.engine as engine
import animeippo.recommendation.scoring as scoring
from animeippo.recommendation import dataset
import pandas as pd
import pytest

from tests import test_data


class ProviderStub:
    def __init__(
        self,
        seasonal=test_data.FORMATTED_MAL_SEASONAL_LIST,
        user=test_data.FORMATTED_MAL_USER_LIST,
        cache=None,
    ):
        self.seasonal = seasonal
        self.user = user
        self.cache = cache

    def get_seasonal_anime_list(self, *args, **kwargs):
        return pd.DataFrame(self.seasonal).set_index("id")

    def get_user_anime_list(self, *args, **kwargs):
        return pd.DataFrame(self.user).set_index("id")

    def get_related_anime(self, *args, **kwargs):
        return pd.DataFrame()

    def get_features(self, *args, **kwargs):
        return ["genres"]


def test_recommend_seasonal_anime_for_user_by_genre():
    provider = ProviderStub()
    data = dataset.UserDataSet(
        provider.get_user_anime_list(), provider.get_seasonal_anime_list(), provider.get_features()
    )
    data.features = provider.get_features()

    scorer = scoring.FeaturesSimilarityScorer(data.features)
    recengine = engine.AnimeRecommendationEngine()

    recengine.add_scorer(scorer)

    recommendations = recengine.fit_predict(data)

    assert recommendations["title"].tolist() == [
        "Shingeki no Kyojin: The Fake Season",
        "Golden Kamuy 4th Season",
    ]


def test_multiple_scorers_can_be_added():
    provider = ProviderStub()
    data = dataset.UserDataSet(
        provider.get_user_anime_list(), provider.get_seasonal_anime_list(), provider.get_features()
    )

    data.features = provider.get_features()

    scorer = scoring.FeaturesSimilarityScorer(data.features)
    scorer2 = scoring.StudioCountScorer()
    recengine = engine.AnimeRecommendationEngine()

    recengine.add_scorer(scorer)
    recengine.add_scorer(scorer2)

    recommendations = recengine.fit_predict(data)

    assert recommendations["title"].tolist() == [
        "Shingeki no Kyojin: The Fake Season",
        "Golden Kamuy 4th Season",
    ]


def test_runtime_error_is_raised_when_no_scorers_exist():
    recengine = engine.AnimeRecommendationEngine()

    data = dataset.UserDataSet(None, None)

    with pytest.raises(RuntimeError):
        recengine.fit_predict(data)
