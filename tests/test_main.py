from animeippo import main
from animeippo.recommendation.engine import AnimeRecommendationEngine
from tests.recommendation.test_recommendation import ProviderStub
from tests.test_data import FORMATTED_ANI_SEASONAL_LIST


def test_engine_can_be_created():
    recommender = main.create_recommender()

    assert recommender is not None
    assert recommender.__class__ == AnimeRecommendationEngine


def test_user_dataset_can_be_created():
    data = main.create_user_dataset("Janiskeisari", "2023", "winter", ProviderStub())

    assert data.seasonal is not None
    assert data.watchlist is not None


def test_user_dataset_creation_skips_related_anime_if_it_already_exists():
    data = main.create_user_dataset(
        "Janiskeisari", "2023", "spring", ProviderStub(seasonal=FORMATTED_ANI_SEASONAL_LIST)
    )
    assert len(data.seasonal["related_anime"]) > 0
