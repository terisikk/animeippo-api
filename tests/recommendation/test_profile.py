import pandas as pd
import pytest

from animeippo.recommendation import profile, encoding, dataset

from tests import test_provider, test_data


def test_profile_analyser_can_run():
    profiler = profile.ProfileAnalyser(test_provider.AsyncProviderStub())
    profiler.encoder = encoding.CategoricalEncoder()

    categories = profiler.analyse("Janiskeisari")

    assert len(categories) > 0


def test_nswf_tags_can_be_removed():
    df = pl.DataFrame({"title": ["Test"], "tags": [["test1", "test2"]], "nsfw_tags": [["test1"]]})

    profiler = profile.ProfileAnalyser(test_provider.AsyncProviderStub())

    assert profiler.get_nsfw_tags(df) == ["test1"]


@pytest.mark.asyncio
async def test_profile_analyser_can_run_when_async_loop_is_already_running():
    profiler = profile.ProfileAnalyser(test_provider.AsyncProviderStub())
    profiler.encoder = encoding.CategoricalEncoder()

    categories = profiler.analyse("Janiskeisari")

    assert len(categories) > 0


def test_user_profile_can_be_constructred():
    watchlist = pl.DataFrame(test_data.FORMATTED_MAL_USER_LIST)

    encoder = encoding.CategoricalEncoder()
    encoder.fit(["Action", "Adventure"], "features")

    watchlist["encoded"] = encoder.encode(watchlist)

    user_profile = profile.UserProfile("Test", watchlist)

    assert user_profile.watchlist is not None
    assert user_profile.genre_correlations is not None
    assert user_profile.studio_correlations is not None
    assert user_profile.director_correlations is not None

    assert user_profile.get_cluster_correlations() is not None
    assert user_profile.get_feature_correlations(["Action", "Adventure"]) is not None


def test_user_profile_can_be_constructred_with_no_watchlist():
    user_profile = profile.UserProfile("Test", None)

    assert user_profile.watchlist is None
    assert user_profile.genre_correlations is None
    assert user_profile.studio_correlations is None
    assert user_profile.director_correlations is None


def test_user_profile_can_be_constructed_with_missing_data():
    watchlist = pl.DataFrame(test_data.FORMATTED_MAL_USER_LIST)
    watchlist = watchlist.drop("cluster", axis=1)

    user_profile = profile.UserProfile("Test", watchlist)

    assert user_profile.watchlist is not None
    assert user_profile.get_cluster_correlations() is None
    assert user_profile.get_feature_correlations(["Action", "Adventure"]) is None


def test_user_top_genres_and_tags_can_be_categorized():
    data = pl.DataFrame(test_data.FORMATTED_ANI_SEASONAL_LIST)
    data["score"] = [10, 8]

    uprofile = profile.UserProfile("Test", data)
    dset = dataset.RecommendationModel(uprofile, None)

    profiler = profile.ProfileAnalyser(None)

    categories = profiler.get_categories(dset)

    assert len(categories) > 0

    uprofile = profile.UserProfile("Test", data)
    uprofile.genre_correlations = pl.Series([1, 1], index=["Absurd", "Nonexisting"])
    dset = dataset.RecommendationModel(uprofile, None)

    categories = profiler.get_categories(dset)
    assert len(categories) > 0

    data = data.drop("genres", axis=1)
    uprofile = profile.UserProfile("Test", data)

    dset = dataset.RecommendationModel(uprofile, None)

    categories = profiler.get_categories(dset)
    assert len(categories) > 0


def test_clusters_can_be_categorized():
    data = pl.DataFrame(test_data.FORMATTED_ANI_SEASONAL_LIST)
    data["cluster"] = [0, 1]

    uprofile = profile.UserProfile("Test", data)
    dset = dataset.RecommendationModel(uprofile, None)

    profiler = profile.ProfileAnalyser(None)

    categories = profiler.get_cluster_categories(dset)

    assert len(categories) > 0
