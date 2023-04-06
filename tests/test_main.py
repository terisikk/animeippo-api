from animeippo import main
from animeippo.recommendation.engine import AnimeRecommendationEngine


def test_engine_can_be_created():
    recommender = main.create_recommender()

    assert recommender is not None
    assert recommender.__class__ == AnimeRecommendationEngine
