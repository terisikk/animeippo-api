from animeippo.recommendation import engine, model, scoring, categories, profile, encoding
from animeippo.clustering import model as clustering
import polars as pl
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

    async def get_seasonal_anime_list(self, *args, **kwargs):
        return pl.DataFrame(self.seasonal)

    async def get_user_anime_list(self, *args, **kwargs):
        return pl.DataFrame(self.user)

    async def get_related_anime(self, *args, **kwargs):
        return pl.DataFrame()

    async def get_feature_names(self, *args, **kwargs):
        return ["genres"]

    async def get_user_manga_list(self, *args, **kwargs):
        return pl.DataFrame()

    def get_nsfw_tags(self, *args, **kwargs):
        return []


@pytest.mark.asyncio
async def test_recommend_seasonal_anime_for_user_by_genre():
    provider = ProviderStub()
    data = model.RecommendationModel(
        profile.UserProfile("Test", await provider.get_user_anime_list()),
        await provider.get_seasonal_anime_list(),
    )

    scorer = scoring.FeaturesSimilarityScorer()
    recengine = engine.AnimeRecommendationEngine(
        clustering.AnimeClustering(), encoding.CategoricalEncoder()
    )

    recengine.add_scorer(scorer)

    recommendations = recengine.fit_predict(data)

    assert recommendations["title"].to_list() == [
        "Shingeki no Kyojin: The Fake Season",
        "Copper Kamuy 4th Season",
    ]


@pytest.mark.asyncio
async def test_multiple_scorers_can_be_added():
    provider = ProviderStub()
    data = model.RecommendationModel(
        profile.UserProfile("Test", await provider.get_user_anime_list()),
        await provider.get_seasonal_anime_list(),
    )

    scorer = scoring.FeaturesSimilarityScorer()
    scorer2 = scoring.StudioCountScorer()
    recengine = engine.AnimeRecommendationEngine(
        clustering.AnimeClustering(), encoding.CategoricalEncoder()
    )

    recengine.add_scorer(scorer)
    recengine.add_scorer(scorer2)

    recommendations = recengine.fit_predict(data)

    assert recommendations["title"].to_list() == [
        "Shingeki no Kyojin: The Fake Season",
        "Copper Kamuy 4th Season",
    ]


def test_runtime_error_is_raised_when_dataset_is_empty():
    recengine = engine.AnimeRecommendationEngine(
        clustering.AnimeClustering(), encoding.CategoricalEncoder()
    )

    data = model.RecommendationModel(None, None)

    with pytest.raises(RuntimeError):
        recengine.fit_predict(data)


def test_runtime_error_is_raised_when_no_scorers_exist():
    recengine = engine.AnimeRecommendationEngine(
        clustering.AnimeClustering(), encoding.CategoricalEncoder()
    )

    with pytest.raises(RuntimeError):
        recengine.score_anime(model.RecommendationModel(None, None))


def test_categorize():
    recengine = engine.AnimeRecommendationEngine(
        clustering.AnimeClustering(), encoding.CategoricalEncoder()
    )
    recengine.add_categorizer(categories.ContinueWatchingCategory())
    recengine.add_categorizer(categories.StudioCategory())
    recengine.add_categorizer(categories.GenreCategory(100))

    data = model.RecommendationModel(
        profile.UserProfile("Test", pl.DataFrame(test_data.FORMATTED_MAL_USER_LIST)), None, None
    )

    data.recommendations = pl.DataFrame(
        {
            "id": [1],
            "popularityscore": [1],
            "continuationscore": [2],
            "sourcescore": [3],
            "directscore": [4],
            "clusterscore": [5],
            "formatscore": [1],
            "studiocorrelationscore": [6],
            "cluster": [1],
            "features": [["test"]],
            "source": ["original"],
            "score": [123],
            "user_status": [None],
            "recommend_score": [1],
            "final_score": [1],
        }
    )

    cats = recengine.categorize_anime(data)

    assert len(cats) > 0
    assert cats[0].get("name", False)
    assert cats[1].get("items", False)
