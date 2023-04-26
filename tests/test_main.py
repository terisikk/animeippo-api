from animeippo import main
from animeippo.recommendation.engine import AnimeRecommendationEngine
from tests.recommendation.test_recommendation import ProviderStub


def test_engine_can_be_created():
    recommender = main.create_recommender([])

    assert recommender is not None
    assert recommender.__class__ == AnimeRecommendationEngine


def test_user_dataset_can_be_created():
    data = main.create_user_dataset("Janiskeisari", "2023", "winter", ProviderStub())

    assert data.seasonal is not None
    assert data.watchlist is not None
