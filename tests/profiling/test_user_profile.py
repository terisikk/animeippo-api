import polars as pl

from animeippo.profiling.model import UserProfile
from tests import test_data


def test_user_profile_can_be_constructred():
    watchlist = pl.DataFrame(test_data.FORMATTED_MAL_USER_LIST)

    user_profile = UserProfile("Test", watchlist)

    assert user_profile.watchlist is not None
    assert user_profile.genre_correlations is not None
    assert user_profile.studio_correlations is not None
    assert user_profile.director_correlations is not None


def test_user_profile_can_be_constructred_with_no_watchlist():
    user_profile = UserProfile("Test", None)

    assert user_profile.watchlist is None
    assert user_profile.genre_correlations is None
    assert user_profile.studio_correlations is None
    assert user_profile.director_correlations is None


def test_correlations_are_consistent():
    data = pl.DataFrame(test_data.FORMATTED_ANI_SEASONAL_LIST)
    data = data.with_columns(score=pl.Series([10.0, 8.0]))

    uprofile = UserProfile("Test", data)

    previous = uprofile.get_director_correlations()

    for _ in range(0, 10):
        actual = uprofile.get_director_correlations()
        assert actual.item(0, "weight") == previous.item(0, "weight")
        assert actual.item(0, "name") == previous.item(0, "name")
        assert actual is not None
        previous = actual


def test_favourite_source_calculation():
    data = pl.DataFrame(test_data.FORMATTED_ANI_USER_LIST)
    data = data.with_columns(score=pl.Series([10.0, 8.0]))

    uprofile = UserProfile("Test", data)

    assert uprofile.get_favourite_source() == "original"


def test_favourite_source_calculation_defaults_to_manga():
    data = pl.DataFrame(
        {
            "title": ["Anime 1", "Anime 2", "Anime 3"],
            "score": [9.0, 8.5, 8.0],
            "source": [None, None, None],
        }
    )

    uprofile = UserProfile("Test", data)

    assert uprofile.get_favourite_source() == "manga"
