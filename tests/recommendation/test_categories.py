import pandas as pd

from animeippo.recommendation import categories, dataset


def test_most_popular_category():
    cat = categories.MostPopularCategory()

    recommendations = pd.DataFrame(
        {"popularityscore": [2, 3, 1], "title": ["Test 1", "Test 2", "Test 3"]}
    )

    data = dataset.UserDataSet(None, None, None)
    data.recommendations = recommendations

    actual = cat.categorize(data)

    assert actual["title"].tolist() == ["Test 2", "Test 1", "Test 3"]


def test_continue_watching_category():
    cat = categories.ContinueWatchingCategory()

    recommendations = pd.DataFrame(
        {
            "continuationscore": [0, 0, 1],
            "user_status": ["in_progress", "completed", "in_progress"],
            "title": ["Test 1", "Test 2", "Test 3"],
        }
    )

    data = dataset.UserDataSet(None, None, None)
    data.recommendations = recommendations

    actual = cat.categorize(data)

    assert actual["title"].tolist() == ["Test 3"]


def test_source_category():
    cat = categories.SourceCategory()

    watchlist = pd.DataFrame({"source": ["Manga", "Light_Novel", "Other"], "score": [10, 9, 8]})

    recommendations = pd.DataFrame(
        {
            "directscore": [1, 2, 3],
            "title": ["Test 1", "Test 2", "Test 3"],
            "source": ["Manga", "Manga", "Ligh_Novel"],
        }
    )

    data = dataset.UserDataSet(watchlist, None, None)
    data.recommendations = recommendations

    actual = cat.categorize(data)

    assert actual["title"].tolist() == ["Test 2", "Test 1"]


def test_source_category_descriptions():
    cat = categories.SourceCategory()

    watchlist = pd.DataFrame({"source": ["Manga", "Original", "Other"], "score": [10, 9, 8]})

    recommendations = pd.DataFrame(
        {
            "directscore": [1, 2, 3],
            "title": ["Test 1", "Test 2", "Test 3"],
            "source": ["Manga", "Other", "Original"],
        }
    )

    data = dataset.UserDataSet(watchlist, None, None)
    data.recommendations = recommendations

    cat.categorize(data)

    assert cat.description == "Based on a Manga"

    data.watchlist = pd.DataFrame({"source": ["Manga", "Original", "Other"], "score": [9, 10, 8]})
    cat.categorize(data)

    assert cat.description == "Anime Originals"

    data.watchlist = pd.DataFrame({"source": ["Manga", "Original", "Other"], "score": [9, 8, 10]})
    cat.categorize(data)

    assert cat.description == "Unusual Sources"


def test_studio_category():
    cat = categories.StudioCategory()

    recommendations = pd.DataFrame(
        {"studioaveragescore": [1, 3, 2], "title": ["Test 1", "Test 2", "Test 3"]}
    )

    data = dataset.UserDataSet(None, None, None)
    data.recommendations = recommendations

    actual = cat.categorize(data)

    assert actual["title"].tolist() == ["Test 2", "Test 3", "Test 1"]


def test_cluster_category():
    cat = categories.ClusterCategory(0)

    watchlist = pd.DataFrame(
        {
            "features": [
                ["Action", "Sports", "Romance"],
                ["Action", "Romance"],
                ["Sports", "Comedy"],
            ],
            "cluster": [0, 0, 1],
        }
    )

    recommendations = pd.DataFrame(
        {
            "cluster": [0, 1, 1],
            "title": ["Test 1", "Test 2", "Test 3"],
        }
    )

    data = dataset.UserDataSet(watchlist, None, None)
    data.recommendations = recommendations

    actual = cat.categorize(data)

    assert actual["title"].tolist() == ["Test 1"]
    assert cat.description == "Action Romance"

    actual = cat.categorize(data, 0)

    assert actual["title"].tolist() == []

    data.recommendations = pd.DataFrame(
        {
            "cluster": [1, 1, 1],
            "title": ["Test 1", "Test 2", "Test 3"],
        }
    )

    actual = cat.categorize(data)

    assert len(actual) == 0


def test_cluster_category_returns_none_if_not_enough_clusters():
    cat = categories.ClusterCategory(5)

    watchlist = pd.DataFrame(
        {
            "features": [
                ["Action", "Sports", "Romance"],
                ["Action", "Romance"],
                ["Sports", "Comedy"],
            ],
            "cluster": [0, 0, 1],
        }
    )

    recommendations = pd.DataFrame(
        {
            "cluster": [0, 1, 1],
            "title": ["Test 1", "Test 2", "Test 3"],
        }
    )

    data = dataset.UserDataSet(watchlist, None, None)
    data.recommendations = recommendations

    actual = cat.categorize(data)

    assert actual is None


def test_your_top_picks_category():
    cat = categories.YourTopPicks()

    recommendations = pd.DataFrame(
        {
            "status": ["releasing", "finished", "cancelled"],
            "user_status": [None, None, None],
            "title": ["Test 1", "Test 2", "Test 3"],
            "recommend_score": [0, 1, 2],
            "continuationscore": [0, 0, 0],
        }
    )

    data = dataset.UserDataSet(None, None, None)
    data.recommendations = recommendations

    actual = cat.categorize(data)

    assert actual["title"].tolist() == ["Test 2", "Test 1"]


def test_top_upcoming_category():
    cat = categories.TopUpcoming()

    recommendations = pd.DataFrame(
        {
            "status": ["not_yet_released", "finished", "cancelled"],
            "user_status": [None, None, None],
            "title": ["Test 1", "Test 2", "Test 3"],
            "recommend_score": [0, 1, 2],
            "continuationscore": [0, 0, 0],
        }
    )

    data = dataset.UserDataSet(None, None, None)
    data.recommendations = recommendations

    actual = cat.categorize(data)

    assert actual["title"].tolist() == ["Test 1"]


def test_because_you_liked():
    cat = categories.BecauseYouLiked(0)

    user_data = pd.DataFrame(
        {
            "score": [1, 2],
            "encoded": [[1, 1], [0, 1]],
            "user_complete_date": [1, 2],
            "title": ["W1", "W2"],
        }
    )

    recommendations = pd.DataFrame(
        {"title": ["Test 1", "Test 2", "Test 3"], "encoded": [[0, 1], [1, 0], [0, 0]]}
    )

    data = dataset.UserDataSet(user_data, None, None)
    data.recommendations = recommendations

    actual = cat.categorize(data)

    assert actual.to_list() == [1, 0, 0]


def test_because_you_liked_does_not_fail_with_empty_likes():
    cat = categories.BecauseYouLiked(99)

    user_data = pd.DataFrame(
        {
            "score": [1, 1],
            "user_complete_date": [1, 2],
        }
    )

    data = dataset.UserDataSet(user_data, None, None)

    actual = cat.categorize(data)

    assert actual is None
