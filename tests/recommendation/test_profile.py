import polars as pl
import pytest

from animeippo.recommendation import model, profile, encoding

from tests import test_provider, test_data


def test_profile_analyser_can_run():
    profiler = profile.ProfileAnalyser(test_provider.AsyncProviderStub())
    profiler.encoder = encoding.CategoricalEncoder()

    categories = profiler.analyse("Janiskeisari")

    assert len(categories) > 0


@pytest.mark.asyncio
async def test_profile_analyser_can_run_when_async_loop_is_already_running():
    profiler = profile.ProfileAnalyser(test_provider.AsyncProviderStub())
    profiler.encoder = encoding.CategoricalEncoder()

    categories = profiler.analyse("Janiskeisari")

    assert len(categories) > 0


def test_user_profile_can_be_constructred():
    watchlist = pl.DataFrame(test_data.FORMATTED_MAL_USER_LIST)

    user_profile = profile.UserProfile("Test", watchlist)

    assert user_profile.watchlist is not None
    assert user_profile.genre_correlations is not None
    assert user_profile.studio_correlations is not None
    assert user_profile.director_correlations is not None


def test_user_profile_can_be_constructred_with_no_watchlist():
    user_profile = profile.UserProfile("Test", None)

    assert user_profile.watchlist is None
    assert user_profile.genre_correlations is None
    assert user_profile.studio_correlations is None
    assert user_profile.director_correlations is None


def test_user_top_genres_and_tags_can_be_categorized(mocker):
    data = pl.DataFrame(test_data.FORMATTED_ANI_SEASONAL_LIST)
    data = data.with_columns(score=pl.Series([10.0, 8.0]))

    uprofile = profile.UserProfile("Test", data)
    dset = model.RecommendationModel(uprofile, None)

    profiler = profile.ProfileAnalyser(None)

    categories = profiler.get_categories(dset)

    assert len(categories) > 0

    uprofile = profile.UserProfile("Test", data)
    uprofile.genre_correlations = pl.DataFrame(
        {"weight": [1, 1], "name": ["Absurd", "Nonexisting"]}
    )
    dset = model.RecommendationModel(uprofile, None)

    mocker.patch(
        "animeippo.recommendation.analysis.weight_categoricals_correlation",
        return_value=pl.DataFrame({"weight": [1, 1], "name": ["Absurd", "Nonexisting"]}),
    )

    categories = profiler.get_categories(dset)
    assert len(categories) > 0

    data = data.drop("genres")
    uprofile = profile.UserProfile("Test", data)

    dset = model.RecommendationModel(uprofile, None)

    categories = profiler.get_categories(dset)
    assert len(categories) > 0


def test_clusters_can_be_categorized():
    data = pl.DataFrame(test_data.FORMATTED_ANI_SEASONAL_LIST)
    data = data.with_columns(cluster=pl.Series([0, 1]))

    uprofile = profile.UserProfile("Test", data)
    dset = model.RecommendationModel(uprofile, None)

    profiler = profile.ProfileAnalyser(None)

    categories = profiler.get_cluster_categories(dset)

    assert len(categories) > 0


def test_correlations_are_consistent():
    data = pl.DataFrame(test_data.FORMATTED_ANI_SEASONAL_LIST)
    data = data.with_columns(score=pl.Series([10.0, 8.0]))

    uprofile = profile.UserProfile("Test", data)

    previous = uprofile.get_director_correlations()

    for i in range(0, 10):
        actual = uprofile.get_director_correlations()
        assert actual.item(0, "weight") == previous.item(0, "weight")
        assert actual.item(0, "name") == previous.item(0, "name")
        assert actual is not None
        previous = actual
