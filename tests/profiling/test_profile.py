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


def test_profile_analyser_can_run_with_seasonal():
    profiler = analyser.ProfileAnalyser(test_provider.AsyncProviderStub())
    profiler.encoder = encoding.CategoricalEncoder()

    categories = profiler.analyse("Janiskeisari", year="2023", season="SPRING")

    assert len(categories) > 0


@pytest.mark.asyncio
async def test_profile_analyser_seasonal_works_with_running_loop():
    profiler = analyser.ProfileAnalyser(test_provider.AsyncProviderStub())
    profiler.encoder = encoding.CategoricalEncoder()

    categories = profiler.analyse("Janiskeisari", year="2023", season="SPRING")

    assert len(categories) > 0


@pytest.mark.asyncio
async def test_profile_analyser_can_run_when_async_loop_is_already_running():
    profiler = analyser.ProfileAnalyser(test_provider.AsyncProviderStub())
    profiler.encoder = encoding.CategoricalEncoder()

    categories = profiler.analyse("Janiskeisari")

    assert len(categories) > 0


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

        def get_tag_lookup(self):
            return {
                1: {"name": "Shounen", "category": "Demographic", "isAdult": False},
                2: {"name": "Super Power", "category": "Theme-Fantasy", "isAdult": False},
                3: {"name": "Mythology", "category": "Theme-Fantasy", "isAdult": False},
                4: {"name": "Gore", "category": "Theme-Other", "isAdult": False},
            }

        def get_genres(self):
            return {"Action", "Adventure", "Drama", "Fantasy", "Supernatural", "Comedy", "Sci-Fi"}

    data = pl.DataFrame(test_data.FORMATTED_ANI_SEASONAL_LIST)
    data = data.with_columns(
        cluster=pl.Series([0, 1]),
        score=pl.Series([8.0, 9.0]),
        user_status=pl.Series(["COMPLETED", "CURRENT"]),
    )

    uprofile = UserProfile("Test", data)
    dset = RecommendationModel(uprofile, None)

    profiler = analyser.ProfileAnalyser(ProviderStub())

    categories = profiler.get_cluster_categories(dset)

    assert len(categories) > 0
    assert "stats" in categories[0]
    assert "count" in categories[0]["stats"]
    assert "mean_score" in categories[0]["stats"]
    assert "completion_rate" in categories[0]["stats"]


def test_seasonal_recommendations_added_to_clusters():
    class ProviderStub:
        def get_nsfw_tags(self):
            return []

        def get_tag_lookup(self):
            return {
                1: {"name": "Shounen", "category": "Demographic", "isAdult": False},
                2: {"name": "Super Power", "category": "Theme-Fantasy", "isAdult": False},
            }

        def get_genres(self):
            return {"Action", "Adventure", "Drama", "Fantasy"}

        async def get_seasonal_anime_list(self, year, season):
            return pl.DataFrame(
                {
                    "id": [100, 200, 300],
                    "title": ["Seasonal 1", "Seasonal 2", "Unmatchable"],
                    "genres": [["Action", "Fantasy"], ["Drama"], []],
                    "features": [["Action", "Fantasy", "Shounen"], ["Drama"], []],
                    "ranks": [
                        {"Action": 75, "Fantasy": 75, "Shounen": 90},
                        {"Drama": 75},
                        {},
                    ],
                    "continuation_to": [[999], [], []],
                }
            )

    data = pl.DataFrame(test_data.FORMATTED_ANI_SEASONAL_LIST)
    data = data.with_columns(
        cluster=pl.Series([0, 1]),
        score=pl.Series([8.0, 9.0]),
        user_status=pl.Series(["COMPLETED", "CURRENT"]),
    )

    uprofile = UserProfile("Test", data)
    dset = RecommendationModel(uprofile, None)

    profiler = analyser.ProfileAnalyser(ProviderStub())
    profiler.profile = uprofile
    profiler.profile.watchlist = data

    # Fit encoder and clusterer like databuilder would
    all_features = data.explode("features")["features"].unique().drop_nulls()
    profiler.encoder.fit(all_features)
    data = data.with_columns(encoded=profiler.encoder.encode(data))
    data = data.with_columns(cluster=profiler.clusterer.cluster_by_features(data))
    profiler.profile.watchlist = data

    categories = profiler.get_cluster_categories(dset)
    profiler.add_seasonal_recommendations(categories, "2026", None)

    has_recs = any("recommendations" in cat for cat in categories)
    assert has_recs
    assert profiler.seasonal is not None


def test_filter_seasonal_without_continuation_column():
    profiler = analyser.ProfileAnalyser(None)
    profiler.profile = UserProfile("Test", pl.DataFrame({"id": [1], "user_status": ["COMPLETED"]}))

    seasonal = pl.DataFrame({"id": [100], "title": ["Test"]})
    result = profiler.filter_seasonal(seasonal)

    assert len(result) == 1


def test_seasonal_recommendations_without_continuation_column():
    class ProviderStub:
        def get_nsfw_tags(self):
            return []

        def get_tag_lookup(self):
            return {1: {"name": "Shounen", "category": "Demographic", "isAdult": False}}

        def get_genres(self):
            return {"Action"}

        async def get_seasonal_anime_list(self, year, season):
            return pl.DataFrame(
                {
                    "id": [100],
                    "title": ["Seasonal 1"],
                    "genres": [["Action"]],
                    "features": [["Action", "Shounen"]],
                    "ranks": [{"Action": 75, "Shounen": 90}],
                }
            )

    data = pl.DataFrame(test_data.FORMATTED_ANI_SEASONAL_LIST)
    data = data.with_columns(
        cluster=pl.Series([0, 1]),
        score=pl.Series([8.0, 9.0]),
        user_status=pl.Series(["COMPLETED", "CURRENT"]),
    )

    profiler = analyser.ProfileAnalyser(ProviderStub())
    profiler.profile = UserProfile("Test", data)

    all_features = data.explode("features")["features"].unique().drop_nulls()
    profiler.encoder.fit(all_features)
    data = data.with_columns(encoded=profiler.encoder.encode(data))
    data = data.with_columns(cluster=profiler.clusterer.cluster_by_features(data))
    profiler.profile.watchlist = data

    categories = profiler.get_cluster_categories(RecommendationModel(profiler.profile, None))
    profiler.add_seasonal_recommendations(categories, "2026", None)

    assert profiler.seasonal is not None


def test_seasonal_recommendations_handles_none_seasonal():
    class ProviderStub:
        def get_nsfw_tags(self):
            return []

        def get_tag_lookup(self):
            return {}

        def get_genres(self):
            return set()

        async def get_seasonal_anime_list(self, year, season):
            return None

    profiler = analyser.ProfileAnalyser(ProviderStub())
    categories = [{"name": "Test", "items": [1]}]
    profiler.add_seasonal_recommendations(categories, "2026", None)

    assert profiler.seasonal is None
    assert "recommendations" not in categories[0]


def test_clusters_can_be_categorized_with_nsfw_filtering():
    """Test that NSFW tags are correctly filtered when categorizing clusters.

    This test ensures that get_nsfw_tags() returns string tag names (not IDs)
    that can be properly filtered from the features column.
    """

    class ProviderStub:
        def get_nsfw_tags(self):
            # Must return tag names (strings), not IDs (integers)
            return {"Bondage", "Hentai", "Explicit Sex"}

        def get_tag_lookup(self):
            return {
                1: {"name": "Action", "category": "Theme-Action", "isAdult": False},
                2: {"name": "Bondage", "category": "Sexual Content", "isAdult": True},
                3: {"name": "Fantasy", "category": "Theme-Fantasy", "isAdult": False},
                4: {"name": "Comedy", "category": "Theme-Comedy", "isAdult": False},
                5: {"name": "Romance", "category": "Theme-Romance", "isAdult": False},
            }

        def get_genres(self):
            return {"Action", "Comedy", "Fantasy", "Romance"}

    # Create data with features that include some NSFW tags
    data = pl.DataFrame(
        {
            "id": [1, 2],
            "title": ["Anime 1", "Anime 2"],
            "features": [["Action", "Bondage", "Fantasy"], ["Comedy", "Romance"]],
            "cluster": [0, 1],
            "score": [8.0, 9.0],
            "user_status": ["COMPLETED", "CURRENT"],
        }
    )

    uprofile = UserProfile("Test", data)
    dset = RecommendationModel(uprofile, None)

    profiler = analyser.ProfileAnalyser(ProviderStub())

    # This should not raise a type error about List(Int64) vs Categorical
    categories = profiler.get_cluster_categories(dset)

    assert len(categories) > 0
