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

    actual = cat.categorize(data)

    assert actual["title"].to_list() == ["Test 2", "Test 1", "Test 3"]


def test_continue_watching_category():
    cat = categories.ContinueWatchingCategory()

    recommendations = pl.DataFrame(
        {
            "continuationscore": [0, 0, 1],
            "final_score": [0, 0, 1],
            "user_status": ["in_progress", "completed", "in_progress"],
            "title": ["Test 1", "Test 2", "Test 3"],
        }
    )

    data = RecommendationModel(None, None, None)
    data.recommendations = recommendations

    actual = cat.categorize(data)

    assert actual["title"].to_list() == ["Test 3"]


def test_source_category():
    cat = categories.SourceCategory()

    watchlist = pl.DataFrame({"source": ["manga", "light_novel", "other"], "score": [10, 9, 8]})

    recommendations = pl.DataFrame(
        {
            "directscore": [1, 2, 3],
            "final_score": [1, 2, 3],
            "title": ["Test 1", "Test 2", "Test 3"],
            "source": ["manga", "manga", "ligh_novel"],
            "user_status": [None, None, None],
        }
    )

    uprofile = UserProfile("Test", watchlist)
    data = RecommendationModel(uprofile, None, None)
    data.recommendations = recommendations

    actual = cat.categorize(data)

    assert actual["title"].to_list() == ["Test 2", "Test 1"]


def test_source_category_defaults_to_manga_without_scores():
    cat = categories.SourceCategory()

    watchlist = pl.DataFrame(
        {"source": ["manga", "light_novel", "other"], "score": [None, None, None]}
    )

    recommendations = pl.DataFrame(
        {
            "directscore": [1, 2, 3],
            "final_score": [1, 2, 3],
            "title": ["Test 1", "Test 2", "Test 3"],
            "source": ["manga", "manga", "ligh_novel"],
            "user_status": [None, None, None],
        }
    )

    uprofile = UserProfile("Test", watchlist)
    data = RecommendationModel(uprofile, None, None)
    data.recommendations = recommendations

    actual = cat.categorize(data)

    assert actual["title"].to_list() == ["Test 2", "Test 1"]


def test_source_category_descriptions():
    cat = categories.SourceCategory()

    watchlist = pl.DataFrame({"source": ["manga", "original", "other"], "score": [10, 9, 8]})

    recommendations = pl.DataFrame(
        {
            "directscore": [1, 2, 3],
            "final_score": [1, 2, 3],
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
            "final_score": [1, 3, 2],
            "title": ["Test 1", "Test 2", "Test 3"],
            "user_status": [None, None, None],
        }
    )

    data = RecommendationModel(None, None, None)
    data.recommendations = recommendations

    actual = cat.categorize(data)

    assert actual["title"].to_list() == ["Test 2", "Test 3", "Test 1"]


def test_your_top_picks_category():
    cat = categories.YourTopPicksCategory()

    recommendations = pl.DataFrame(
        {
            "title": ["Test 1", "Test 2", "Test 3"],
            "user_status": [None, None, None],
            "status": ["releasing", "releasing", "releasing"],
            "continuationscore": [0, 0, 10],
            "final_score": [2, 3, 1],
        }
    )

    data = RecommendationModel(None, None, None)
    data.recommendations = recommendations

    actual = cat.categorize(data)

    assert actual["title"].to_list() == ["Test 2", "Test 1"]


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
            "final_score": [0, 1, 2, 3],
            "continuationscore": [0, 0, 0, 0],
        }
    )

    data = RecommendationModel(None, None, None)
    data.recommendations = recommendations

    actual = cat.categorize(data)

    assert actual["title"].to_list() == ["Test 1", "Test 4"]


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

    actual = cat.categorize(data)["id"].to_list()

    assert actual == [3, 5, 4]
    assert cat.description == "Because You Liked W2"


def test_because_you_liked_does_not_raise_error_with_empty_likes():
    cat = categories.BecauseYouLikedCategory(99)

    watchlist = pl.DataFrame(
        {"score": [1, 1], "user_complete_date": [1, 2], "user_status": [None, None]}
    )

    uprofile = UserProfile("Test", watchlist)
    data = RecommendationModel(uprofile, None, None)
    data.recommendations = watchlist

    actual = cat.categorize(data)

    assert actual is None


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

    actual = cat.categorize(data)

    assert actual is None


def test_simulcastscategory(mocker):
    cat = categories.SimulcastsCategory()

    recommendations = pl.DataFrame(
        {
            "season_year": [2022, 2022, 2022],
            "season": ["summer", "summer", "winter"],
            "title": ["Test 1", "Test 2", "Test 3"],
            "final_score": [0, 1, 2],
            "continuationscore": [0, 0, 0],
        }
    )

    mocker.patch(
        "animeippo.meta.meta.get_current_anime_season",
        return_value=(2022, "summer"),
    )

    data = RecommendationModel(None, None, None)
    data.recommendations = recommendations

    actual = cat.categorize(data)

    assert actual["title"].to_list() == ["Test 2", "Test 1"]


def test_adapatation_category():
    cat = categories.AdaptationCategory()

    recommendations = pl.DataFrame(
        {
            "adaptationscore": [0, 1, 1],
            "final_score": [0, 0, 1],
            "user_status": ["in_progress", "completed", "in_progress"],
            "title": ["Test 1", "Test 2", "Test 3"],
        }
    )

    data = RecommendationModel(None, None, None)
    data.recommendations = recommendations

    actual = cat.categorize(data)

    assert actual["title"].to_list() == ["Test 3"]


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
            "final_score": [1, 1, 1],
            "title": ["Test 1", "Test 2", "Test 3"],
            "user_status": [None, None, None],
        }
    )

    uprofile = UserProfile("Test", watchlist)
    data = RecommendationModel(uprofile, None, None)
    data.recommendations = recommendations

    actual = cat.categorize(data)

    assert actual["title"].to_list() == ["Test 1", "Test 3"]
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

    actual = cat.categorize(data)

    assert actual is None


def test_genre_category_can_cache_values():
    cat = categories.GenreCategory(0)

    recommendations = pl.DataFrame(
        {
            "genres": [["Action", "Fantasy"], ["Drama"], ["Action"]],
            "discourage_score": [1, 1, 1],
            "final_score": [1, 1, 1],
            "title": ["Test 1", "Test 2", "Test 3"],
            "user_status": [None, None, None],
        }
    )

    uprofile = UserProfile("Test", None)
    uprofile.genre_correlations = pl.DataFrame({"weight": [0.5, 0.5], "name": ["Action", "Sports"]})

    data = RecommendationModel(uprofile, None, None)
    data.recommendations = recommendations

    actual = cat.categorize(data)

    assert actual["title"].to_list() == ["Test 1", "Test 3"]
    assert cat.description == "Action"


def test_discourage_wrapper():
    cat = categories.YourTopPicksCategory()
    dcat = categories.DiscouragingWrapper(cat)

    recommendations = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "title": ["Test 1", "Test 2", "Test 3"],
            "user_status": [None, None, None],
            "status": ["releasing", "releasing", "releasing"],
            "continuationscore": [0.0, 0.0, 0.0],
            "recommend_score": [2.0, 2.1, 1.99],
            "final_score": [2.0, 2.1, 1.99],
            "discourage_score": [1.0, 1.0, 1.0],
        }
    )

    data = RecommendationModel(None, None, None)
    data.recommendations = recommendations

    actual = dcat.categorize(data, max_items=2)

    assert actual["title"].to_list() == ["Test 2", "Test 1"]
    assert data.recommendations["discourage_score"].to_list() == [0.75, 0.75, 1]

    actual = dcat.categorize(data, max_items=2)

    assert actual["title"].to_list() == ["Test 3", "Test 2"]

    assert dcat.description == cat.description
    assert data.recommendations["discourage_score"].to_list() == [0.75, 0.5, 0.75]


def test_debug_category_returns_all_recommendations():
    cat = categories.DebugCategory()

    recommendations = pl.DataFrame(
        {
            "title": ["Test 1", "Test 2", "Test 3"],
            "final_score": [2.0, 2.1, 1.99],
        }
    )

    data = RecommendationModel(None, None, None)
    data.recommendations = recommendations

    assert cat.categorize(data)["title"].to_list() == ["Test 2", "Test 1", "Test 3"]


def test_planning_category():
    cat = categories.PlanningCategory()

    recommendations = pl.DataFrame(
        {
            "title": ["Test 1", "Test 2", "Test 3"],
            "user_status": [None, "planning", "in_progress"],
            "final_score": [1, 1, 1],
        }
    )

    data = RecommendationModel(None, None, None)
    data.recommendations = recommendations

    actual = cat.categorize(data)

    assert actual["title"].to_list() == ["Test 2"]
