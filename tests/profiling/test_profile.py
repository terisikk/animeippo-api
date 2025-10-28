import polars as pl
import pytest

from animeippo.analysis import encoding
from animeippo.profiling import analyser
from animeippo.profiling.model import UserProfile
from animeippo.recommendation.model import RecommendationModel
from tests import test_data, test_provider


def test_profile_analyser_can_run():
    profiler = analyser.ProfileAnalyser(test_provider.AsyncProviderStub())
    profiler.encoder = encoding.CategoricalEncoder()

    categories = profiler.analyse("Janiskeisari")

    assert len(categories) > 0


@pytest.mark.asyncio
async def test_profile_analyser_can_run_when_async_loop_is_already_running():
    profiler = analyser.ProfileAnalyser(test_provider.AsyncProviderStub())
    profiler.encoder = encoding.CategoricalEncoder()

    categories = profiler.analyse("Janiskeisari")

    assert len(categories) > 0


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


def test_user_top_genres_and_tags_can_be_categorized(mocker):
    data = pl.DataFrame(test_data.FORMATTED_ANI_SEASONAL_LIST)
    data = data.with_columns(score=pl.Series([10.0, 8.0]))

    uprofile = UserProfile("Test", data)
    dset = RecommendationModel(uprofile, None)

    profiler = analyser.ProfileAnalyser(None)

    categories = profiler.get_categories(dset)

    assert len(categories) > 0

    uprofile = UserProfile("Test", data)
    uprofile.genre_correlations = pl.DataFrame(
        {"weight": [1, 1], "name": ["Absurd", "Nonexisting"]}
    )
    dset = RecommendationModel(uprofile, None)

    mocker.patch(
        "animeippo.analysis.statistics.weight_categoricals_correlation",
        return_value=pl.DataFrame({"weight": [1, 1], "name": ["Absurd", "Nonexisting"]}),
    )

    categories = profiler.get_categories(dset)
    assert len(categories) > 0

    data = data.drop("genres")
    uprofile = UserProfile("Test", data)

    dset = RecommendationModel(uprofile, None)

    categories = profiler.get_categories(dset)
    assert len(categories) > 0

    data = data.drop("tags")
    uprofile = UserProfile("Test", data)

    dset = RecommendationModel(uprofile, None)

    categories = profiler.get_categories(dset)
    assert len(categories) == 0


def test_clusters_can_be_categorized():
    class ProviderStub:
        def get_nsfw_tags(self):
            return []

    data = pl.DataFrame(test_data.FORMATTED_ANI_SEASONAL_LIST)
    data = data.with_columns(cluster=pl.Series([0, 1]))

    uprofile = UserProfile("Test", data)
    dset = RecommendationModel(uprofile, None)

    profiler = analyser.ProfileAnalyser(ProviderStub())

    categories = profiler.get_cluster_categories(dset)

    assert len(categories) > 0


def test_clusters_can_be_categorized_with_nsfw_filtering():
    """Test that NSFW tags are correctly filtered when categorizing clusters.

    This test ensures that get_nsfw_tags() returns string tag names (not IDs)
    that can be properly filtered from the features column.
    """

    class ProviderStub:
        def get_nsfw_tags(self):
            # Must return tag names (strings), not IDs (integers)
            return {"Bondage", "Hentai", "Explicit Sex"}

    # Create data with features that include some NSFW tags
    data = pl.DataFrame(
        {
            "id": [1, 2],
            "title": ["Anime 1", "Anime 2"],
            "features": [["Action", "Bondage", "Fantasy"], ["Comedy", "Romance"]],
            "cluster": [0, 1],
            "score": [8.0, 9.0],
        }
    )

    uprofile = UserProfile("Test", data)
    dset = RecommendationModel(uprofile, None)

    profiler = analyser.ProfileAnalyser(ProviderStub())

    # This should not raise a type error about List(Int64) vs Categorical
    categories = profiler.get_cluster_categories(dset)

    assert len(categories) > 0


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
