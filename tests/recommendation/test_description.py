import polars as pl
import pytest

from animeippo.profiling.model import UserProfile
from animeippo.providers.anilist.data import ALL_FEATURES, ALL_GENRES
from animeippo.recommendation import description
from animeippo.recommendation.model import RecommendationModel
from tests import test_data


@pytest.fixture
def describer():
    describer = description.AnimeDescriber(ALL_GENRES)
    describer.fit()
    return describer


@pytest.fixture
def recommendation_model():
    user_profile = UserProfile("Test", pl.DataFrame(test_data.FORMATTED_ANI_USER_LIST))

    # Normally we would fit user profile, but let's just add pre-selected correlations
    user_profile.feature_correlations = pl.DataFrame({"name": list(ALL_FEATURES), "weight": 0})

    seasonal = pl.DataFrame(test_data.FORMATTED_ANI_SEASONAL_LIST)

    data = RecommendationModel(user_profile=user_profile, seasonal=seasonal)
    return data


def test_describe_shows_with_various_combinations(describer, recommendation_model):
    recommendations = pl.DataFrame(
        {
            "id": [123],
            "title": ["Chic Kawa"],
            "feature_info": [
                [
                    {
                        "name": "Magic",
                        "rank": 100,
                        "category": "Theme-Fantasy",
                        "mood": None,
                        "intensity": None,
                        "spoiler": False,
                    }
                ],
            ],
        }
    )

    recommendation_model.recommendations = recommendations

    assert describer.describe_shows(recommendation_model).row(0)[1] == ["Magic"]
