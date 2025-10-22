import polars as pl

from animeippo.profiling.model import UserProfile
from animeippo.recommendation import categories
from animeippo.recommendation.model import RecommendationModel


def test_most_popular_category():
    cat = categories.MostPopularCategory()

    recommendations = pl.DataFrame(
        {"popularityscore": [2, 3, 1], "title": ["Test 1", "Test 2", "Test 3"]}
    )

    data = RecommendationModel(None, None, None)
    data.recommendations = recommendations

    mask, sorting_info = cat.categorize(data)

    result = recommendations.filter(mask).sort(**sorting_info)
    assert result["title"].to_list() == ["Test 2", "Test 1", "Test 3"]


def test_continue_watching_category():
    cat = categories.ContinueWatchingCategory()

    recommendations = pl.DataFrame(
        {
            "continuationscore": [0, 0, 1],
            "recommend_score": [0, 0, 1],
            "user_status": ["in_progress", "completed", "in_progress"],
            "title": ["Test 1", "Test 2", "Test 3"],
        }
    )

    data = RecommendationModel(None, None, None)
    data.recommendations = recommendations

    mask, sorting_info = cat.categorize(data)

    result = recommendations.filter(mask).sort(**sorting_info)
    assert result["title"].to_list() == ["Test 3"]


def test_source_category():
    cat = categories.SourceCategory()

    watchlist = pl.DataFrame({"source": ["manga", "light_novel", "other"], "score": [10, 9, 8]})

    recommendations = pl.DataFrame(
        {
            "directscore": [1, 2, 3],
            "adjusted_score": [1, 2, 3],
            "title": ["Test 1", "Test 2", "Test 3"],
            "source": ["manga", "manga", "ligh_novel"],
            "user_status": [None, None, None],
        }
    )

    uprofile = UserProfile("Test", watchlist)
    data = RecommendationModel(uprofile, None, None)
    data.recommendations = recommendations

    mask, sorting_info = cat.categorize(data)

    result = recommendations.filter(mask).sort(**sorting_info)
    assert result["title"].to_list() == ["Test 2", "Test 1"]


def test_source_category_defaults_to_manga_without_scores():
    cat = categories.SourceCategory()

    watchlist = pl.DataFrame(
        {"source": ["manga", "light_novel", "other"], "score": [None, None, None]}
    )

    recommendations = pl.DataFrame(
        {
            "directscore": [1, 2, 3],
            "adjusted_score": [1, 2, 3],
            "title": ["Test 1", "Test 2", "Test 3"],
            "source": ["manga", "manga", "ligh_novel"],
            "user_status": [None, None, None],
        }
    )

    uprofile = UserProfile("Test", watchlist)
    data = RecommendationModel(uprofile, None, None)
    data.recommendations = recommendations

    mask, sorting_info = cat.categorize(data)

    result = recommendations.filter(mask).sort(**sorting_info)
    assert result["title"].to_list() == ["Test 2", "Test 1"]


def test_source_category_descriptions():
    cat = categories.SourceCategory()

    watchlist = pl.DataFrame({"source": ["manga", "original", "other"], "score": [10, 9, 8]})

    recommendations = pl.DataFrame(
        {
            "directscore": [1, 2, 3],
            "adjusted_score": [1, 2, 3],
            "title": ["Test 1", "Test 2", "Test 3"],
            "source": ["manga", "other", "original"],
            "user_status": [None, None, None],
        }
    )

    uprofile = UserProfile("Test", watchlist)
    data = RecommendationModel(uprofile, None, None)
    data.recommendations = recommendations

    cat.categorize(data)

    assert cat.description == "Based on a Manga"

    data.watchlist = pl.DataFrame({"source": ["manga", "original", "other"], "score": [9, 10, 8]})
    cat.categorize(data)

    assert cat.description == "Anime Originals"

    data.watchlist = pl.DataFrame({"source": ["manga", "original", "other"], "score": [9, 8, 10]})
    cat.categorize(data)

    assert cat.description == "Unusual Sources"


def test_studio_category():
    cat = categories.StudioCategory()

    recommendations = pl.DataFrame(
        {
            "studiocorrelationscore": [1, 3, 2],
            "formatscore": [1, 3, 2],
            "recommend_score": [1, 3, 2],
            "title": ["Test 1", "Test 2", "Test 3"],
            "user_status": [None, None, None],
        }
    )

    data = RecommendationModel(None, None, None)
    data.recommendations = recommendations

    mask, sorting_info = cat.categorize(data)

    result = recommendations.filter(mask).sort(**sorting_info)
    assert result["title"].to_list() == ["Test 2", "Test 3", "Test 1"]


def test_your_top_picks_category():
    cat = categories.YourTopPicksCategory()

    recommendations = pl.DataFrame(
        {
            "title": ["Test 1", "Test 2", "Test 3"],
            "user_status": [None, None, None],
            "status": ["releasing", "releasing", "releasing"],
            "continuationscore": [0, 0, 10],
            "recommend_score": [2, 3, 1],
        }
    )

    data = RecommendationModel(None, None, None)
    data.recommendations = recommendations

    mask, sorting_info = cat.categorize(data)

    result = recommendations.filter(mask).sort(**sorting_info)
    assert result["title"].to_list() == ["Test 2", "Test 1"]


def test_top_upcoming_category(mocker):
    cat = categories.TopUpcomingCategory()

    # patch get_current_anime_season
    mocker.patch(
        "animeippo.meta.meta.get_current_anime_season",
        return_value=(2022, "spring"),
    )

    recommendations = pl.DataFrame(
        {
            "status": ["not_yet_released", "finished", "cancelled", "not_yet_released"],
            "user_status": [None, None, None, None],
            "season_year": [2022, 2022, 2022, 2023],
            "season": ["summer", "winter", "summer", "spring"],
            "title": ["Test 1", "Test 2", "Test 3", "Test 4"],
            "recommend_score": [0, 1, 2, 3],
            "continuationscore": [0, 0, 0, 0],
        }
    )

    data = RecommendationModel(None, None, None)
    data.recommendations = recommendations

    mask, sorting_info = cat.categorize(data)

    result = recommendations.filter(mask).sort(**sorting_info)
    assert result["title"].to_list() == ["Test 1", "Test 4"]


def test_because_you_liked():
    cat = categories.BecauseYouLikedCategory(0, distance_metric="cosine")

    watchlist = pl.DataFrame(
        {
            "id": [1, 2],
            "score": [1, 2],
            "encoded": [[1, 1], [0, 1]],
            "user_complete_date": [1, 2],
            "title": ["W1", "W2"],
        },
    )

    watchlist = watchlist.with_columns(id=pl.col("id").cast(pl.UInt32))

    recommendations = pl.DataFrame(
        {
            "id": [3, 4, 5],
            "title": ["Test 1", "Test 2", "Test 3"],
            "encoded": [[0, 1], [1, 0], [0, 0]],
            "season_year": [1, 1, 1],
            "season": ["a", "a", "a"],
            "user_status": [None, None, None],
        }
    )

    recommendations = recommendations.with_columns(id=pl.col("id").cast(pl.UInt32))

    uprofile = UserProfile("Test", watchlist)
    data = RecommendationModel(uprofile, None, None)
    data.recommendations = recommendations
    data.similarity_matrix = pl.DataFrame(
        {
            "3": [0.5, 1.0],
            "4": [0.5, 0.0],
            "5": [0.0, 0.5],
            "id": [1, 2],
        }
    )

    mask, sorting_info = cat.categorize(data)

    result = recommendations.filter(mask).sort(**sorting_info)
    assert result["id"].to_list() == [3, 5, 4]
    assert cat.description == "Because You Liked W2"


def test_because_you_liked_does_not_raise_error_with_empty_likes():
    cat = categories.BecauseYouLikedCategory(99)

    watchlist = pl.DataFrame(
        {"score": [1, 1], "user_complete_date": [1, 2], "user_status": [None, None]}
    )

    uprofile = UserProfile("Test", watchlist)
    data = RecommendationModel(uprofile, None, None)
    data.recommendations = watchlist

    mask, sorting_info = cat.categorize(data)

    assert mask is False
    assert sorting_info == {}


def test_because_you_liked_does_not_raise_error_with_missing_similarity():
    cat = categories.BecauseYouLikedCategory(0)

    watchlist = pl.DataFrame(
        {
            "id": [1, 2],
            "score": [1, 2],
            "encoded": [[1, 1], [0, 1]],
            "user_complete_date": [1, 2],
            "title": ["W1", "W2"],
        }
    )

    uprofile = UserProfile("Test", watchlist)
    data = RecommendationModel(uprofile, None, None)

    data.similarity_matrix = pl.DataFrame(
        {
            "3": [0.5, 1.0],
            "4": [0.5, 0.0],
            "5": [0.0, 0.5],
            "id": [6, 7],
        }
    )

    mask, sorting_info = cat.categorize(data)

    assert mask is False
    assert sorting_info == {}


def test_simulcastscategory(mocker):
    cat = categories.SimulcastsCategory()

    recommendations = pl.DataFrame(
        {
            "season_year": [2022, 2022, 2022],
            "season": ["summer", "summer", "winter"],
            "title": ["Test 1", "Test 2", "Test 3"],
            "recommend_score": [0, 1, 2],
            "continuationscore": [0, 0, 0],
        }
    )

    mocker.patch(
        "animeippo.meta.meta.get_current_anime_season",
        return_value=(2022, "summer"),
    )

    data = RecommendationModel(None, None, None)
    data.recommendations = recommendations

    mask, sorting_info = cat.categorize(data)

    result = recommendations.filter(mask).sort(**sorting_info)
    assert result["title"].to_list() == ["Test 2", "Test 1"]


def test_adapatation_category():
    cat = categories.AdaptationCategory()

    recommendations = pl.DataFrame(
        {
            "adaptationscore": [0, 1, 1],
            "recommend_score": [0, 0, 1],
            "user_status": ["in_progress", "completed", "in_progress"],
            "title": ["Test 1", "Test 2", "Test 3"],
        }
    )

    data = RecommendationModel(None, None, None)
    data.recommendations = recommendations

    mask, sorting_info = cat.categorize(data)

    result = recommendations.filter(mask).sort(**sorting_info)
    assert result["title"].to_list() == ["Test 3"]


def test_genre_category():
    cat = categories.GenreCategory(0)

    watchlist = pl.DataFrame(
        {
            "genres": [
                ["Action", "Sports", "Romance"],
                ["Action", "Romance"],
                ["Sports", "Comedy"],
            ],
            "user_status": ["not_watched", None, "in_progress"],
            "score": [10, 10, 10],
        }
    )

    recommendations = pl.DataFrame(
        {
            "genres": [["Action", "Fantasy"], ["Drama"], ["Action"]],
            "discourage_score": [1, 1, 1],
            "adjusted_score": [1, 1, 1],
            "recommend_score": [1, 1, 1],
            "title": ["Test 1", "Test 2", "Test 3"],
            "user_status": [None, None, None],
        }
    )

    uprofile = UserProfile("Test", watchlist)
    data = RecommendationModel(uprofile, None, None)
    data.recommendations = recommendations

    mask, sorting_info = cat.categorize(data)

    result = recommendations.filter(mask).sort(**sorting_info)
    assert result["title"].to_list() == ["Test 1", "Test 3"]
    assert cat.description == "Action"


def test_genre_category_returns_none_for_too_big_genre_number():
    cat = categories.GenreCategory(99)

    watchlist = pl.DataFrame(
        {
            "genres": [
                ["Action", "Sports", "Romance"],
                ["Action", "Romance"],
                ["Sports", "Comedy"],
            ],
            "user_status": ["not_watched", None, "in_progress"],
            "score": [10, 10, 10],
        }
    )

    uprofile = UserProfile("Test", watchlist)
    data = RecommendationModel(uprofile, None, None)

    mask, sorting_info = cat.categorize(data)

    assert mask is False
    assert sorting_info == {}


def test_genre_category_can_cache_values():
    cat = categories.GenreCategory(0)

    recommendations = pl.DataFrame(
        {
            "genres": [["Action", "Fantasy"], ["Drama"], ["Action"]],
            "discourage_score": [1, 1, 1],
            "adjusted_score": [1, 1, 1],
            "recommend_score": [1, 1, 1],
            "title": ["Test 1", "Test 2", "Test 3"],
            "user_status": [None, None, None],
        }
    )

    uprofile = UserProfile("Test", None)
    uprofile.genre_correlations = pl.DataFrame({"weight": [0.5, 0.5], "name": ["Action", "Sports"]})

    data = RecommendationModel(uprofile, None, None)
    data.recommendations = recommendations

    mask, sorting_info = cat.categorize(data)

    result = recommendations.filter(mask).sort(**sorting_info)
    assert result["title"].to_list() == ["Test 1", "Test 3"]
    assert cat.description == "Action"


def test_ranking_orchestrator_diversity_adjustment():
    from animeippo.recommendation.ranking import RankingOrchestrator

    recommendations = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "title": ["Test 1", "Test 2", "Test 3"],
            "recommend_score": [2.0, 2.1, 1.99],
        }
    )

    orchestrator = RankingOrchestrator()

    # Initialize diversity adjustment
    orchestrator.diversity_adjustment = pl.DataFrame(
        {
            "id": recommendations["id"],
            "diversity_adjustment": 0,
        }
    )

    # First call - should return items in order of recommend_score
    result_ids = orchestrator.adjust_by_diversity(recommendations, top_n=2)

    assert result_ids == [2, 1]  # IDs of Test 2 and Test 1
    assert orchestrator.diversity_adjustment["diversity_adjustment"].to_list() == [0.25, 0.25, 0]

    # Second call - items 1 and 2 are discouraged, so Test 3 should come up
    result_ids = orchestrator.adjust_by_diversity(recommendations, top_n=2)

    assert result_ids == [3, 2]  # Test 3 (id=3) now scores higher, Test 2 (id=2) still second
    assert orchestrator.diversity_adjustment["diversity_adjustment"].to_list() == [0.25, 0.5, 0.25]


def test_ranking_orchestrator_render_with_diversity_adjusted_categories():
    from animeippo.recommendation.ranking import RankingOrchestrator

    watchlist = pl.DataFrame(
        {
            "genres": [["Action", "Drama"], ["Action", "Comedy"], ["Action"]],
            "user_status": ["completed", "completed", "completed"],
            "score": [10, 9, 10],
        }
    )

    recommendations = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "title": ["Action 1", "Action 2", "Drama 1", "Comedy 1"],
            "genres": [["Action"], ["Action"], ["Drama"], ["Comedy"]],
            "recommend_score": [2.0, 1.9, 1.8, 1.7],
            "user_status": [None, None, None, None],
        }
    )

    uprofile = UserProfile("Test", watchlist)
    data = RecommendationModel(uprofile, None, None)
    data.recommendations = recommendations

    # Create genre categories that will be diversity-adjusted
    # GenreCategory picks genre by correlation score
    genre_cat_1 = categories.GenreCategory(0)  # First genre
    genre_cat_2 = categories.GenreCategory(0)  # Same genre again

    orchestrator = RankingOrchestrator()
    result = orchestrator.render(data, [genre_cat_1, genre_cat_2])

    # Both categories should be rendered
    assert len(result) == 2
    assert result[0]["name"] == result[1]["name"]  # Same genre

    # Both should have items (diversity adjustment applied)
    assert len(result[0]["items"]) > 0
    assert len(result[1]["items"]) > 0


def test_ranking_orchestrator_render_with_non_diversity_adjusted_categories():
    from animeippo.recommendation.ranking import RankingOrchestrator

    recommendations = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "title": ["Test 1", "Test 2", "Test 3"],
            "recommend_score": [2.0, 2.1, 1.99],
            "popularityscore": [1.0, 2.0, 3.0],
        }
    )

    data = RecommendationModel(None, None, None)
    data.recommendations = recommendations

    # MostPopularCategory is not in diversity_adjusted_categories
    popular_cat = categories.MostPopularCategory()

    orchestrator = RankingOrchestrator()
    result = orchestrator.render(data, [popular_cat])

    assert len(result) == 1
    assert result[0]["name"] == "Most Popular for This Year"
    # Should return top 25 items (or all if less than 25)
    assert result[0]["items"] == [3, 2, 1]  # Sorted by popularityscore descending


def test_ranking_orchestrator_render_skips_empty_categories():
    from animeippo.recommendation.ranking import RankingOrchestrator

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
            "recommend_score": [2.0, 1.0],
        }
    )

    uprofile = UserProfile("Test", watchlist)
    data = RecommendationModel(uprofile, None, None)
    data.recommendations = recommendations

    # BecauseYouLikedCategory will return (False, {}) if no liked items
    empty_cat = categories.BecauseYouLikedCategory(99)

    orchestrator = RankingOrchestrator()
    result = orchestrator.render(data, [empty_cat])

    # Should skip the empty category
    assert len(result) == 0


def test_debug_category_returns_all_recommendations():
    cat = categories.DebugCategory()

    recommendations = pl.DataFrame(
        {
            "title": ["Test 1", "Test 2", "Test 3"],
            "recommend_score": [2.0, 2.1, 1.99],
        }
    )

    data = RecommendationModel(None, None, None)
    data.recommendations = recommendations

    mask, sorting_info = cat.categorize(data)

    result = recommendations.filter(mask).sort(**sorting_info)
    assert result["title"].to_list() == ["Test 2", "Test 1", "Test 3"]


def test_planning_category():
    cat = categories.PlanningCategory()

    recommendations = pl.DataFrame(
        {
            "title": ["Test 1", "Test 2", "Test 3"],
            "user_status": [None, "planning", "in_progress"],
            "recommend_score": [1, 1, 1],
        }
    )

    data = RecommendationModel(None, None, None)
    data.recommendations = recommendations

    mask, sorting_info = cat.categorize(data)

    result = recommendations.filter(mask).sort(**sorting_info)
    assert result["title"].to_list() == ["Test 2"]
