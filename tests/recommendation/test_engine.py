from animeippo.recommendation import engine, scoring, dataset, categories
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

    scorer = scoring.FeaturesSimilarityScorer()
    recengine = engine.AnimeRecommendationEngine()

    recengine.add_scorer(scorer)

    recommendations = recengine.fit_predict(data)

    assert recommendations["title"].tolist() == [
        "Shingeki no Kyojin: The Fake Season",
        "Copper Kamuy 4th Season",
    ]


def test_multiple_scorers_can_be_added():
    provider = ProviderStub()
    data = dataset.UserDataSet(
        provider.get_user_anime_list(), provider.get_seasonal_anime_list(), provider.get_features()
    )

    scorer = scoring.FeaturesSimilarityScorer()
    scorer2 = scoring.StudioCountScorer()
    recengine = engine.AnimeRecommendationEngine()

    recengine.add_scorer(scorer)
    recengine.add_scorer(scorer2)

    recommendations = recengine.fit_predict(data)

    assert recommendations["title"].tolist() == [
        "Shingeki no Kyojin: The Fake Season",
        "Copper Kamuy 4th Season",
    ]


def test_runtime_error_is_raised_when_dataset_is_empty():
    recengine = engine.AnimeRecommendationEngine()

    data = dataset.UserDataSet(None, None)

    with pytest.raises(RuntimeError):
        recengine.fit_predict(data)


def test_runtime_error_is_raised_when_no_scorers_exist():
    recengine = engine.AnimeRecommendationEngine()

    with pytest.raises(RuntimeError):
        recengine.score_anime(None, None)


def test_categorize():
    recengine = engine.AnimeRecommendationEngine()
    recengine.add_categorizer(categories.ContinueWatchingCategory())
    recengine.add_categorizer(categories.ClusterCategory(0))
    recengine.add_categorizer(categories.ClusterCategory(100))

    data = dataset.UserDataSet(pd.DataFrame(test_data.FORMATTED_MAL_USER_LIST), None, None)

    data.recommendations = pd.DataFrame(
        {
            "popularityscore": [1],
            "continuationscore": [2],
            "sourcescore": [3],
            "directscore": [4],
            "clusterscore": [5],
            "studioaveragescore": [6],
            "cluster": [1],
            "features": [["test"]],
            "source": ["Other"],
            "score": [123],
        }
    )

    cats = recengine.categorize_anime(data)

    assert len(cats) > 0
    assert cats[0].get("name", False)
    assert cats[1].get("items", False)
