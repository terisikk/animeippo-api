import pandas as pd
import pytest

from animeippo.recommendation import profile, dataset, encoding

from tests import test_provider


def test_profile_analyser_can_run():
    profiler = profile.ProfileAnalyser(test_provider.AsyncProviderStub())
    profiler.encoder = encoding.CategoricalEncoder()

    categories = profiler.analyse("Janiskeisari")

    assert len(categories) > 0


def test_nswf_tags_can_be_removed():
    df = pd.DataFrame({"title": ["Test"], "tags": [["test1", "test2"]], "nsfw_tags": [["test1"]]})

    profiler = profile.ProfileAnalyser(test_provider.AsyncProviderStub())

    assert profiler.get_nsfw_tags(df) == ["test1"]


@pytest.mark.asyncio
async def test_profile_analyser_can_run_when_async_loop_is_already_running():
    profiler = profile.ProfileAnalyser(test_provider.AsyncProviderStub())
    profiler.encoder = encoding.CategoricalEncoder()

    categories = profiler.analyse("Janiskeisari")

    assert len(categories) > 0


# TODO: This does not test anything ATM?
def test_profile_analyser_splits_watchlist_to_categories():
    profiler = profile.ProfileAnalyser(None)

    watchlist = pd.DataFrame(
        {
            "title": ["Test 1", "Test 2", "Test 3"],
            "features": [
                ["Action", "Sports", "Romance"],
                ["Action", "Romance"],
                ["Sports", "Comedy"],
            ],
            "cluster": [0, 0, 1],
        }
    )

    data = dataset.UserDataSet(watchlist, None, None)

    categories = profiler.get_categories(data)

    print(categories)
