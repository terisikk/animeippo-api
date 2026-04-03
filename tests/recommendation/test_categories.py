import polars as pl

from animeippo.profiling.model import UserProfile
from animeippo.recommendation import categories
from animeippo.recommendation.model import RecommendationModel
from animeippo.recommendation.ranking import RankingOrchestrator


def test_abstract_category_default_behavior():
    class ConcreteCategory(categories.AbstractCategory):
        description = "Test"

        def categorize(self, dataset):
            return super().categorize(dataset)

    cat = ConcreteCategory()
    cat.categorize(None)
    assert cat.needs_diversity is False

    df = pl.DataFrame({"id": [1, 2, 3]})
    assert cat.get_items(df, 2) == [1, 2]


def test_most_popular_category():
    cat = categories.MostPopularCategory()

    recommendations = pl.DataFrame(
        {"popularity": [2, 3, 1], "title": ["Test 1", "Test 2", "Test 3"]}
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
            "discovery_score": [0, 0, 1],
            "user_status": ["in_progress", "COMPLETED", "in_progress"],
            "title": ["Test 1", "Test 2", "Test 3"],
            "format": ["TV", "TV", "TV"],
        }
    )

    data = RecommendationModel(None, None, None)
    data.recommendations = recommendations

    mask, sorting_info = cat.categorize(data)

    result = recommendations.filter(mask).sort(**sorting_info)
    assert result["title"].to_list() == ["Test 3"]


def test_manga_category():
    cat = categories.MangaCategory()

    watchlist = pl.DataFrame({"source": ["MANGA", "LIGHT_NOVEL", "OTHER"], "score": [10, 9, 8]})

    recommendations = pl.DataFrame(
        {
            "directscore": [1, 2, 3],
            "discovery_score": [1, 2, 3],
            "title": ["Test 1", "Test 2", "Test 3"],
            "source": ["MANGA", "MANGA", "LIGHT_NOVEL"],
            "user_status": [None, None, None],
            "format": ["TV", "TV", "TV"],
            "duration": [24, 24, 24],
        }
    )

    uprofile = UserProfile("Test", watchlist)
    data = RecommendationModel(uprofile, None, None)
    data.recommendations = recommendations

    mask, sorting_info = cat.categorize(data)

    result = recommendations.filter(mask).sort(**sorting_info)
    assert result["title"].to_list() == ["Test 2", "Test 1"]


def test_studio_category():
    cat = categories.StudioCategory()

    recommendations = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "studios": [["Bones"], ["Toei"], ["MAPPA"], ["Bones"]],
            "discovery_score": [3, 5, 1, 2],
            "title": ["Test 1", "Test 2", "Test 3", "Test 4"],
            "user_status": [None, None, None, None],
        }
    )

    uprofile = UserProfile("Test", None)
    # MAPPA ranked higher than Bones
    uprofile.studio_correlations = pl.DataFrame({"name": ["MAPPA", "Bones"], "weight": [0.5, 0.3]})

    data = RecommendationModel(uprofile, None, None)
    data.recommendations = recommendations

    mask, sorting_info = cat.categorize(data)

    filtered = recommendations.filter(mask).sort(**sorting_info)
    # Toei excluded
    assert len(filtered) == 3

    # get_items sorts by best studio weight then discovery score
    items = cat.get_items(filtered, top_n=10)
    id_to_title = dict(
        zip(recommendations["id"].to_list(), recommendations["title"].to_list(), strict=True)
    )
    ordered = [id_to_title[i] for i in items]
    # MAPPA (0.5) first, then Bones (0.3) by discovery_score
    assert ordered == ["Test 3", "Test 1", "Test 4"]

    # top_n=None returns all items
    all_items = cat.get_items(filtered, top_n=None)
    assert len(all_items) == 3


def test_studio_category_returns_false_without_correlations():
    cat = categories.StudioCategory()

    uprofile = UserProfile("Test", None)
    uprofile.studio_correlations = None
    data = RecommendationModel(uprofile, None, None)

    mask, sorting_info = cat.categorize(data)

    assert mask is False
    assert sorting_info == {}


def test_studio_category_returns_false_when_all_weights_zero():
    cat = categories.StudioCategory()

    uprofile = UserProfile("Test", None)
    uprofile.studio_correlations = pl.DataFrame(
        {"name": ["StudioA", "StudioB"], "weight": [0.0, -0.1]}
    )
    data = RecommendationModel(uprofile, None, None)

    mask, sorting_info = cat.categorize(data)

    assert mask is False
    assert sorting_info == {}


def test_your_top_picks_category():
    cat = categories.YourTopPicksCategory()

    recommendations = pl.DataFrame(
        {
            "title": ["Test 1", "Test 2", "Test 3"],
            "user_status": [None, None, None],
            "status": ["RELEASING", "RELEASING", "RELEASING"],
            "continuationscore": [0, 0, 10],
            "discovery_score": [2, 3, 1],
            "format": ["TV", "TV", "TV"],
            "duration": [24, 24, 24],
        }
    )

    data = RecommendationModel(None, None, None)
    data.recommendations = recommendations

    mask, sorting_info = cat.categorize(data)

    result = recommendations.filter(mask).sort(**sorting_info)
    assert result["title"].to_list() == ["Test 2", "Test 1"]


def test_top_released_picks_category():
    cat = categories.TopReleasedPicksCategory()

    recommendations = pl.DataFrame(
        {
            "title": ["Test 1", "Test 2", "Test 3", "Test 4", "Test 5"],
            "status": ["RELEASING", "FINISHED", "NOT_YET_RELEASED", "FINISHED", "RELEASING"],
            "user_status": [None, None, None, "COMPLETED", "COMPLETED"],
            "discovery_score": [3, 2, 1, 5, 4],
            "format": ["TV", "TV", "TV", "TV", "TV"],
            "duration": [24, 24, 24, 24, 24],
        }
    )

    data = RecommendationModel(None, None, None)
    data.recommendations = recommendations

    mask, sorting_info = cat.categorize(data)

    result = recommendations.filter(mask).sort(**sorting_info)
    assert result["title"].to_list() == ["Test 1", "Test 2"]


def test_hidden_gems_category():
    cat = categories.HiddenGemsCategory()

    recommendations = pl.DataFrame(
        {
            "title": ["Test 1", "Test 2", "Test 3", "Test 4", "Test 5", "Test 6"],
            "status": ["FINISHED", "RELEASING", "FINISHED", "FINISHED", "FINISHED", "FINISHED"],
            "user_status": [None, None, None, "COMPLETED", None, None],
            "format": ["TV", "TV", "OVA", "TV", "MOVIE", "TV_SHORT"],
            "duration": [24, 24, 30, 24, 90, 5],
            "discovery_score": [8.0, 7.0, 6.0, 9.0, 10.0, 5.0],
            "popularity": [100, 50000, 500, 200, 10, 300],
            "continuationscore": [0, 0, 0, 0, 0, 0],
        }
    )

    data = RecommendationModel(None, None, None)
    data.recommendations = recommendations

    mask, sorting_info = cat.categorize(data)

    # Test 4 excluded (COMPLETED), Test 5 excluded (MOVIE),
    # Test 6 excluded (TV_SHORT, duration < 10)
    # Remaining sorted by discovery_score * (1 - 0.5 * pop_rank)
    result = recommendations.filter(mask).sort(**sorting_info)
    assert result["title"].to_list() == ["Test 1", "Test 3", "Test 2"]


def test_top_movies_category():
    cat = categories.MovieNightCategory()

    recommendations = pl.DataFrame(
        {
            "title": ["Test 1", "Test 2", "Test 3", "Test 4", "Test 5"],
            "format": ["MOVIE", "TV", "MOVIE", "MOVIE", "MOVIE"],
            "user_status": [None, None, "COMPLETED", None, None],
            "status": ["FINISHED", "FINISHED", "FINISHED", "NOT_YET_RELEASED", "FINISHED"],
            "discovery_score": [8.0, 9.0, 10.0, 11.0, 7.0],
        }
    )

    data = RecommendationModel(None, None, None)
    data.recommendations = recommendations

    mask, sorting_info = cat.categorize(data)

    result = recommendations.filter(mask).sort(**sorting_info)
    assert result["title"].to_list() == ["Test 1", "Test 5"]


def test_all_movies_category():
    cat = categories.AllMoviesCategory()

    recommendations = pl.DataFrame(
        {
            "title": ["Movie A", "TV Show", "Movie B", "Movie C"],
            "format": ["MOVIE", "TV", "MOVIE", "MOVIE"],
            "user_status": [None, None, "COMPLETED", None],
            "discovery_score": [0.8, 0.9, 0.7, 0.6],
        }
    )

    data = RecommendationModel(None, None, None)
    data.recommendations = recommendations

    mask, sorting_info = cat.categorize(data)

    result = recommendations.filter(mask).sort(**sorting_info)
    # Includes all movies except completed, sorted by discovery_score
    assert result["title"].to_list() == ["Movie A", "Movie C"]


def test_top_upcoming_category(mocker):
    cat = categories.TopUpcomingCategory()

    # patch get_current_anime_season
    mocker.patch(
        "animeippo.meta.meta.get_current_anime_season",
        return_value=(2022, "SPRING"),
    )

    recommendations = pl.DataFrame(
        {
            "status": ["NOT_YET_RELEASED", "FINISHED", "CANCELLED", "NOT_YET_RELEASED"],
            "user_status": [None, None, None, None],
            "season_year": [2022, 2022, 2022, 2023],
            "season": ["SUMMER", "WINTER", "SUMMER", "SPRING"],
            "title": ["Test 1", "Test 2", "Test 3", "Test 4"],
            "discovery_score": [0, 1, 2, 3],
            "continuationscore": [0, 0, 0, 0],
            "format": ["TV", "TV", "TV", "TV"],
            "duration": [24, 24, 24, 24],
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
            "season": ["SUMMER", "SUMMER", "WINTER"],
            "title": ["Test 1", "Test 2", "Test 3"],
            "discovery_score": [0, 1, 2],
            "continuationscore": [0, 0, 0],
            "format": ["TV", "TV", "TV"],
            "duration": [24, 24, 24],
        }
    )

    mocker.patch(
        "animeippo.meta.meta.get_current_anime_season",
        return_value=(2022, "SUMMER"),
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
            "discovery_score": [0, 0, 1],
            "user_status": ["in_progress", "COMPLETED", "in_progress"],
            "title": ["Test 1", "Test 2", "Test 3"],
            "format": ["TV", "TV", "TV"],
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
            "discovery_score": [1, 1, 1],
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
            "discovery_score": [1, 1, 1],
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
    orchestrator = RankingOrchestrator([])
    result = orchestrator.adjust_by_diversity([], top_n=5)
    assert result == []


def test_ranking_orchestrator_diversity_adjustment():
    recommendations = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "title": ["Test 1", "Test 2", "Test 3"],
            "discovery_score": [2.0, 2.1, 1.99],
        }
    )

    orchestrator = RankingOrchestrator([])
    orchestrator.recommendations_df = recommendations

    orchestrator.diversity_adjustment = pl.DataFrame(
        {
            "id": recommendations["id"],
            "diversity_adjustment": 0,
        }
    )

    all_ids = recommendations["id"].to_list()

    # First call - should return items in order of discovery_score
    result_ids = orchestrator.adjust_by_diversity(all_ids, top_n=2)

    assert result_ids == [2, 1]
    assert orchestrator.diversity_adjustment["diversity_adjustment"].to_list() == [0.25, 0.25, 0]

    # Second call - items 1 and 2 are discouraged, so Test 3 should come up
    result_ids = orchestrator.adjust_by_diversity(all_ids, top_n=2)

    assert result_ids == [3, 2]
    assert orchestrator.diversity_adjustment["diversity_adjustment"].to_list() == [0.25, 0.5, 0.25]


def test_ranking_orchestrator_render_with_diversity_adjusted_categories():
    watchlist = pl.DataFrame(
        {
            "genres": [["Action", "Drama"], ["Action", "Comedy"], ["Action"]],
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


def test_cluster_category():
    watchlist = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "features": [["Action", "Fantasy"], ["Action", "Fantasy"], ["Drama"], ["Drama"]],
            "cluster": [0, 0, 1, 1],
            "score": [9, 8, 7, 6],
        }
    )

    recommendations = pl.DataFrame(
        {
            "id": [10, 11, 12, 13],
            "title": ["Rec A", "Rec B", "Rec C", "Rec D"],
            "cluster": [0, 0, 1, 1],
            "cluster_similarity": [0.8, 0.7, 0.6, 0.5],
            "discovery_score": [0.9, 0.8, 0.7, 0.6],
        }
    )

    uprofile = UserProfile("Test", watchlist)
    data = RecommendationModel(uprofile, None, None)
    data.recommendations = recommendations

    cat = categories.ClusterCategory(
        nth_cluster=0,
        tag_lookup={},
        genres={"Action", "Fantasy", "Drama"},
    )

    mask, sorting_info = cat.categorize(data)
    result = recommendations.filter(mask).sort(**sorting_info)

    # Should show items from the top-ranked cluster
    assert len(result) > 0
    assert cat.description != "Cluster"


def test_cluster_category_returns_false_when_no_clusters():
    watchlist = pl.DataFrame({"id": [1], "score": [8]})

    uprofile = UserProfile("Test", watchlist)
    data = RecommendationModel(uprofile, None, None)
    data.recommendations = pl.DataFrame({"id": [10], "discovery_score": [0.9]})

    cat = categories.ClusterCategory(nth_cluster=0)
    mask, _ = cat.categorize(data)
    assert mask is False


def test_cluster_category_without_features():
    watchlist = pl.DataFrame({"id": [1, 2, 3], "cluster": [0, 0, 0], "score": [8, 9, 7]})

    recommendations = pl.DataFrame(
        {
            "id": [10, 11, 12],
            "cluster": [0, 0, 0],
            "cluster_similarity": [0.8, 0.7, 0.6],
            "discovery_score": [0.9, 0.8, 0.7],
        }
    )

    uprofile = UserProfile("Test", watchlist)
    data = RecommendationModel(uprofile, None, None)
    data.recommendations = recommendations

    cat = categories.ClusterCategory(nth_cluster=0)
    mask, _ = cat.categorize(data)

    assert mask is not False
    assert "Cluster" in cat.description


def test_cluster_category_returns_false_for_nonexistent_nth():
    watchlist = pl.DataFrame(
        {
            "id": [1, 2],
            "features": [["Action"], ["Action"]],
            "cluster": [0, 0],
            "score": [9, 8],
        }
    )

    recommendations = pl.DataFrame(
        {
            "id": [10],
            "cluster": [0],
            "cluster_similarity": [0.8],
            "discovery_score": [0.9],
        }
    )

    uprofile = UserProfile("Test", watchlist)
    data = RecommendationModel(uprofile, None, None)
    data.recommendations = recommendations

    cat = categories.ClusterCategory(nth_cluster=5, tag_lookup={}, genres={"Action"})
    mask, _ = cat.categorize(data)
    assert mask is False


def test_debug_category_returns_all_recommendations():
    cat = categories.DebugCategory()

    recommendations = pl.DataFrame(
        {
            "title": ["Test 1", "Test 2", "Test 3"],
            "discovery_score": [2.0, 2.1, 1.99],
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
            "title": ["Not planned", "Summer", "Spring"],
            "user_status": [None, "PLANNING", "PLANNING"],
            "season_year": [2026, 2026, 2026],
            "season": ["SPRING", "SUMMER", "SPRING"],
            "discovery_score": [1.0, 0.8, 0.9],
        }
    )

    data = RecommendationModel(None, None, None)
    data.recommendations = recommendations

    mask, sorting_info = cat.categorize(data)

    result = recommendations.filter(mask).sort(**sorting_info)
    assert result["title"].to_list() == ["Spring", "Summer"]


def test_compose_two_pool_lane_pins_strong_continuations():
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "title": ["Strong Cont", "Weak Cont", "Disc 1", "Disc 2", "Disc 3"],
            "continuationscore": [0.9, 0.3, 0.0, 0.0, 0.0],
            "continuationscore_confidence": [0.8, 0.3, 0.0, 0.0, 0.0],
            "discovery_score": [0.5, 0.4, 0.9, 0.8, 0.7],
        }
    )

    result = categories.compose_two_pool_lane(df, continuation_threshold=0.7, max_total=5)

    # Strong continuation pinned first
    assert result[0] == 1
    # Discoveries fill remaining slots
    assert 3 in result
    assert 4 in result


def test_compose_two_pool_lane_interleaves_weak_continuations():
    df = pl.DataFrame(
        {
            "id": list(range(1, 12)),
            "continuationscore": [0.5] + [0.0] * 10,
            "continuationscore_confidence": [0.3] + [0.0] * 10,
            "discovery_score": [0.1] + [1.0 - i * 0.05 for i in range(10)],
        }
    )

    result = categories.compose_two_pool_lane(
        df, continuation_threshold=0.7, weak_interleave_interval=5, max_total=11
    )

    # Weak continuation (id=1) interleaved, not pinned at top
    assert result[0] != 1
    assert 1 in result


def test_compose_two_pool_lane_no_max_total():
    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "continuationscore": [0.0, 0.0, 0.0],
            "continuationscore_confidence": [0.0, 0.0, 0.0],
            "discovery_score": [0.9, 0.8, 0.7],
        }
    )

    result = categories.compose_two_pool_lane(df, max_total=None)
    assert len(result) == 3


def test_compose_two_pool_lane_respects_max_total():
    df = pl.DataFrame(
        {
            "id": list(range(1, 50)),
            "continuationscore": [0.0] * 49,
            "continuationscore_confidence": [0.0] * 49,
            "discovery_score": [1.0 - i * 0.01 for i in range(49)],
        }
    )

    result = categories.compose_two_pool_lane(df, max_total=10)
    assert len(result) == 10


def test_compose_two_pool_lane_with_group_by():
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "group": ["A", "A", "B", "B"],
            "continuationscore": [0.9, 0.0, 0.8, 0.0],
            "continuationscore_confidence": [0.8, 0.0, 0.9, 0.0],
            "discovery_score": [0.5, 0.9, 0.4, 0.7],
        }
    )

    result = categories.compose_two_pool_lane(df, group_by=["group"], max_total=None)

    # Group A comes first, continuation (id=1) pinned
    assert result[0] == 1
    # Group B follows, continuation (id=3) pinned
    group_b = [i for i in result if i in {3, 4}]
    assert group_b[0] == 3

    # Also test with multiple group columns
    df2 = df.with_columns(subgroup=pl.Series(["X", "X", "Y", "Y"]))
    result2 = categories.compose_two_pool_lane(df2, group_by=["group", "subgroup"], max_total=None)
    assert len(result2) == 4

    # Test max_total with groups
    result3 = categories.compose_two_pool_lane(df, group_by=["group"], max_total=2)
    assert len(result3) == 2


def test_simulcasts_category_get_items():
    cat = categories.SimulcastsCategory()

    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "continuationscore": [0.9, 0.0, 0.0],
            "continuationscore_confidence": [0.8, 0.0, 0.0],
            "discovery_score": [0.5, 0.9, 0.7],
        }
    )

    result = cat.get_items(df, top_n=None)
    assert result[0] == 1
    assert len(result) == 3


def test_top_upcoming_category_get_items_respects_season_groups():
    cat = categories.TopUpcomingCategory()

    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "season_year": [2026, 2026, 2026, 2026, 2026],
            "season": ["SPRING", "SPRING", "SPRING", "SUMMER", "SUMMER"],
            "continuationscore": [0.0, 0.8, 0.0, 0.9, 0.0],
            "continuationscore_confidence": [0.0, 0.9, 0.0, 0.8, 0.0],
            "discovery_score": [0.9, 0.5, 0.7, 0.3, 0.8],
        }
    )

    result = cat.get_items(df, top_n=None)

    # Spring items come before summer items
    spring_ids = {1, 2, 3}
    summer_ids = {4, 5}
    last_spring_pos = max(result.index(i) for i in spring_ids)
    first_summer_pos = min(result.index(i) for i in summer_ids)
    assert last_spring_pos < first_summer_pos

    # Within spring, continuation (id=2) is pinned first
    spring_result = [i for i in result if i in spring_ids]
    assert spring_result[0] == 2

    # Within summer, continuation (id=4) is pinned first
    summer_result = [i for i in result if i in summer_ids]
    assert summer_result[0] == 4


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
