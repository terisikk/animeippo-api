import polars as pl

from animeippo.profiling.model import UserProfile
from animeippo.recommendation import categories
from animeippo.recommendation.model import RecommendationModel
from animeippo.recommendation.ranking import RankingOrchestrator, _adjust_by_diversity


def test_ranking_orchestrator_selects_layout_by_data_volume():
    minimal_cat = categories.MostPopularCategory()
    standard_cat = categories.YourTopPicksCategory()
    full_cat = categories.StudioCategory()

    layouts = {
        "minimal": [(minimal_cat, None)],
        "standard": [(standard_cat, None)],
        "full": [(full_cat, None)],
    }

    orchestrator = RankingOrchestrator(layouts)

    # <20 items = minimal
    assert orchestrator.select_layout(10) == layouts["minimal"]
    # 20-100 = standard
    assert orchestrator.select_layout(50) == layouts["standard"]
    # >100 = full
    assert orchestrator.select_layout(200) == layouts["full"]


def test_ranking_orchestrator_skips_categories_below_min_items():
    class HighMinCategory(categories.AbstractCategory):
        description = "Needs Many"

        def __init__(self):
            super().__init__(min_items=5)

        def categorize(self, dataset):
            return True, {"by": "discovery_score", "descending": True}

    orchestrator = RankingOrchestrator([(HighMinCategory(), None)])

    data = RecommendationModel(None, None, None)
    data.recommendations = pl.DataFrame({"id": [1, 2, 3], "discovery_score": [0.9, 0.8, 0.7]})

    result = orchestrator.render(data)
    assert len(result) == 0  # 3 items < min_items=5, skipped


def test_ranking_orchestrator_diversity_adjustment_empty_list():
    diversity_adjustment = pl.DataFrame(
        {"id": pl.Series([], dtype=pl.UInt32), "diversity_adjustment": []}
    )
    result, _ = _adjust_by_diversity(
        [], pl.DataFrame(), diversity_adjustment, top_n=5, discourage_max=0.75
    )
    assert result == []


def test_ranking_orchestrator_diversity_adjustment():
    recommendations = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "title": ["Test 1", "Test 2", "Test 3"],
            "discovery_score": [2.0, 2.1, 1.99],
        }
    )

    diversity_adjustment = pl.DataFrame(
        {
            "id": recommendations["id"],
            "diversity_adjustment": 0,
        }
    )

    all_ids = recommendations["id"].to_list()

    # First call - should return items in order of discovery_score
    result_ids, diversity_adjustment = _adjust_by_diversity(
        all_ids, recommendations, diversity_adjustment, top_n=2, discourage_max=0.75
    )

    assert result_ids == [2, 1]
    assert diversity_adjustment["diversity_adjustment"].to_list() == [0.25, 0.25, 0]

    # Second call - items 1 and 2 are discouraged, so Test 3 should come up
    result_ids, diversity_adjustment = _adjust_by_diversity(
        all_ids, recommendations, diversity_adjustment, top_n=2, discourage_max=0.75
    )

    assert result_ids == [3, 2]
    assert diversity_adjustment["diversity_adjustment"].to_list() == [0.25, 0.5, 0.25]


def test_ranking_orchestrator_render_with_diversity_adjusted_categories():
    watchlist = pl.DataFrame(
        {
            "genres": [["Action", "Drama"], ["Action", "Comedy"], ["Action"]],
            "encoded": [
                {"Action": 1, "Sports": 0, "Romance": 0, "Comedy": 0, "Fantasy": 0, "Drama": 1},
                {"Action": 1, "Sports": 0, "Romance": 1, "Comedy": 1, "Fantasy": 0, "Drama": 0},
                {"Action": 1, "Sports": 0, "Romance": 0, "Comedy": 0, "Fantasy": 0, "Drama": 0},
            ],
            "user_status": ["COMPLETED", "COMPLETED", "COMPLETED"],
            "score": [10, 9, 10],
        }
    )

    recommendations = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "title": ["A1", "A2", "A3", "D1", "D2", "C1"],
            "genres": [["Action"], ["Action"], ["Action"], ["Drama"], ["Drama"], ["Comedy"]],
            "features": [["Action"], ["Action"], ["Action"], ["Drama"], ["Drama"], ["Comedy"]],
            "discovery_score": [2.0, 1.9, 1.8, 1.7, 1.6, 1.5],
            "user_status": [None, None, None, None, None, None],
        }
    )

    uprofile = UserProfile("Test", watchlist)
    uprofile.fit()

    data = RecommendationModel(uprofile, None, None)
    data.recommendations = recommendations

    # Create genre categories that will be diversity-adjusted
    # GenreCategory picks genre by correlation score
    genre_cat_1 = categories.GenreCategory(nth_genre=0, needs_diversity=True)
    genre_cat_2 = categories.GenreCategory(nth_genre=0, needs_diversity=True)

    orchestrator = RankingOrchestrator([(genre_cat_1, None), (genre_cat_2, None)])
    result = orchestrator.render(data)

    # Both categories should be rendered
    assert len(result) == 2
    assert result[0]["name"] == result[1]["name"]  # Same genre

    # Both should have items (diversity adjustment applied)
    assert len(result[0]["items"]) > 0
    assert len(result[1]["items"]) > 0


def test_ranking_orchestrator_render_with_non_diversity_adjusted_categories():
    recommendations = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "title": ["Test 1", "Test 2", "Test 3"],
            "discovery_score": [2.0, 2.1, 1.99],
            "popularity": [1000, 2000, 3000],
        }
    )

    data = RecommendationModel(None, None, None)
    data.recommendations = recommendations

    # MostPopularCategory is not in diversity_adjusted_categories
    popular_cat = categories.MostPopularCategory()

    orchestrator = RankingOrchestrator([(popular_cat, 25)])
    result = orchestrator.render(data)

    assert len(result) == 1
    assert result[0]["name"] == "Most Popular for This Year"
    assert result[0]["items"] == [3, 2, 1]  # Sorted by popularity descending


def test_ranking_orchestrator_render_column_not_found():
    class FakeCategory:
        def categorize(self, dataset):
            dataset.recommendations.select("missing")
            return False, {}

    recommendations = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "title": ["Test 1", "Test 2", "Test 3"],
            "discovery_score": [2.0, 2.1, 1.99],
            "popularity": [1000, 2000, 3000],
        }
    )

    data = RecommendationModel(None, None, None)
    data.recommendations = recommendations

    orchestrator = RankingOrchestrator([(FakeCategory(), 25)])

    # Should not raise
    orchestrator.render(data)


def test_ranking_orchestrator_exclusive_categories_dedupe():
    recommendations = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "title": ["A", "B", "C", "D"],
            "discovery_score": [4.0, 3.0, 2.0, 1.0],
            "popularity": [100, 200, 300, 400],
        }
    )

    data = RecommendationModel(None, None, None)
    data.recommendations = recommendations

    # Two exclusive categories that would normally overlap
    cat1 = categories.MostPopularCategory(exclusive=True)
    cat2 = categories.MostPopularCategory(exclusive=True)

    orchestrator = RankingOrchestrator([(cat1, 2), (cat2, 2)])
    result = orchestrator.render(data)

    assert len(result) == 2
    # First category gets top 2 by popularity
    first_items = set(result[0]["items"])
    second_items = set(result[1]["items"])
    # No overlap between exclusive categories
    assert first_items & second_items == set()


def test_ranking_orchestrator_render_skips_empty_categories():
    watchlist = pl.DataFrame(
        {
            "id": [1, 2],
            "score": [1, 1],
            "user_complete_date": [1, 2],
            "user_status": [None, None],
        }
    )

    recommendations = pl.DataFrame(
        {
            "id": [3, 4],
            "title": ["Test 1", "Test 2"],
            "discovery_score": [2.0, 1.0],
            "features": [["A", "B"], ["A", "B"]],
        }
    )

    uprofile = UserProfile("Test", watchlist)
    data = RecommendationModel(uprofile, None, None)
    data.recommendations = recommendations

    # BecauseYouLikedCategory will return (False, {}) if no liked items
    empty_cat = categories.BecauseYouLikedCategory(99)

    orchestrator = RankingOrchestrator([(empty_cat, 20)])
    result = orchestrator.render(data)

    # Should skip the empty category
    assert len(result) == 0


def test_render_calls_get_items_on_category():
    class CustomCategory(categories.AbstractCategory):
        description = "Custom Lane"

        def categorize(self, dataset):
            return True, {"by": "discovery_score", "descending": True}

        def get_items(self, df, top_n):
            return df.sort("discovery_score", descending=True)["id"].to_list()[:2]

    orchestrator = RankingOrchestrator([(CustomCategory(), None)])

    data = RecommendationModel(None, None, None)
    data.recommendations = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "discovery_score": [0.5, 0.9, 0.7],
        }
    )

    result = orchestrator.render(data)
    assert result[0]["items"] == [2, 3]
