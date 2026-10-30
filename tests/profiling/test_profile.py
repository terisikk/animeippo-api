import polars as pl
import pytest

from animeippo.profiling import analyser
from animeippo.profiling.model import UserProfile
from animeippo.recommendation import recommender_builder
from animeippo.recommendation.model import RecommendationModel
from tests import test_data, test_provider


@pytest.mark.asyncio
async def test_profile_analyser_can_run():
    recommender = recommender_builder.build_analysis_recommender("anilist")
    recommender.provider = test_provider.AsyncProviderStub(
        seasonal=test_data.FORMATTED_ANI_SEASONAL_LIST,
        user=test_data.FORMATTED_ANI_USER_LIST,
    )

    profiler = analyser.ProfileAnalyser(recommender)

    _, categories, _ = await profiler.analyse("Janiskeisari")

    assert len(categories) > 0


@pytest.mark.asyncio
async def test_profile_analyser_can_run_with_seasonal():
    recommender = recommender_builder.build_analysis_recommender("anilist")
    recommender.provider = test_provider.AsyncProviderStub(
        seasonal=test_data.FORMATTED_ANI_SEASONAL_LIST,
        user=test_data.FORMATTED_ANI_USER_LIST,
    )

    profiler = analyser.ProfileAnalyser(recommender)

    _, categories, _ = await profiler.analyse("Janiskeisari", year="2023", season="SPRING")

    assert len(categories) > 0


@pytest.mark.asyncio
async def test_profile_analyser_seasonal_works_with_running_loop():
    recommender = recommender_builder.build_analysis_recommender("anilist")
    recommender.provider = test_provider.AsyncProviderStub(
        seasonal=test_data.FORMATTED_ANI_SEASONAL_LIST,
        user=test_data.FORMATTED_ANI_USER_LIST,
    )

    profiler = analyser.ProfileAnalyser(recommender)

    _, categories, _ = await profiler.analyse("Janiskeisari", year="2023", season="SPRING")

    assert len(categories) > 0


@pytest.mark.asyncio
async def test_profile_analyser_can_run_when_async_loop_is_already_running():
    recommender = recommender_builder.build_analysis_recommender("anilist")
    recommender.provider = test_provider.AsyncProviderStub(
        seasonal=test_data.FORMATTED_ANI_SEASONAL_LIST,
        user=test_data.FORMATTED_ANI_USER_LIST,
    )

    profiler = analyser.ProfileAnalyser(recommender)

    _, categories, _ = await profiler.analyse("Janiskeisari")

    assert len(categories) > 0


def test_user_top_genres_and_tags_can_be_categorized(mocker):
    data = pl.DataFrame(test_data.FORMATTED_ANI_SEASONAL_LIST)
    data = data.with_columns(score=pl.Series([10.0, 8.0]))

    uprofile = UserProfile("Test", data)
    dset = RecommendationModel(uprofile, None)

    profiler = analyser.ProfileAnalyser(None)

    categories = profiler.get_profile_top_genre_and_tag_categories(dset)

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

    categories = profiler.get_profile_top_genre_and_tag_categories(dset)
    assert len(categories) > 0

    data = data.drop("genres")
    uprofile = UserProfile("Test", data)

    dset = RecommendationModel(uprofile, None)

    categories = profiler.get_profile_top_genre_and_tag_categories(dset)
    assert len(categories) > 0

    data = data.drop("tags")
    uprofile = UserProfile("Test", data)

    dset = RecommendationModel(uprofile, None)

    categories = profiler.get_profile_top_genre_and_tag_categories(dset)
    assert len(categories) == 0


def test_seasonal_recommendations_added_to_clusters():
    watchlist = pl.DataFrame(test_data.FORMATTED_ANI_USER_LIST)
    watchlist = watchlist.with_columns(
        cluster=pl.Series([0, 1]),
        score=pl.Series([8.0, 9.0]),
        user_status=pl.Series(["COMPLETED", "CURRENT"]),
    )

    recommendations = pl.DataFrame(test_data.FORMATTED_ANI_SEASONAL_LIST)
    recommendations = recommendations.with_columns(
        continuation_to=pl.Series([[30], []]),
        cluster=pl.Series([2, 3]),
        cluster_similarity=pl.Series([0.5, 0.5]),
    )

    recommender = recommender_builder.build_analysis_recommender("anilist")

    profiler = analyser.ProfileAnalyser(recommender)

    categories = profiler.group_recommendations_by_cluster(recommendations, watchlist)

    assert "recommendations" in categories.columns
    # Continuation should have a reassigned cluster
    assert categories.filter(pl.col("cluster") == 0).height == 1


def test_seasonal_recommendations_works_without_continuation_column():
    watchlist = pl.DataFrame(test_data.FORMATTED_ANI_USER_LIST)
    watchlist = watchlist.with_columns(
        cluster=pl.Series([0, 1]),
        score=pl.Series([8.0, 9.0]),
        user_status=pl.Series(["COMPLETED", "CURRENT"]),
    )

    recommendations = pl.DataFrame(test_data.FORMATTED_ANI_SEASONAL_LIST)
    recommendations = recommendations.with_columns(
        cluster=pl.Series([2, 3]),
        cluster_similarity=pl.Series([0.5, 0.5]),
    ).select(pl.exclude("continuation_to"))

    recommender = recommender_builder.build_analysis_recommender("anilist")

    profiler = analyser.ProfileAnalyser(recommender)

    categories = profiler.group_recommendations_by_cluster(recommendations, watchlist)

    assert len(categories) == 2
