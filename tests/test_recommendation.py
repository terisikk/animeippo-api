import animeippo.recommendation.engine as engine
import animeippo.providers.myanimelist as mal

from tests import test_data


def test_recommend_seasonal_anime_for_user_by_genre(requests_mock):
    user = "Janiskeisari"

    url1 = f"{mal.MAL_API_URL}/users/{user}/animelist"
    user_adapter = requests_mock.get(url1, json=test_data.MAL_USER_LIST)  # nosec B113

    year = "2023"
    season = "winter"
    url2 = f"{mal.MAL_API_URL}/anime/season/{year}/{season}"
    season_adapter = requests_mock.get(url2, json=test_data.MAL_SEASONAL_LIST)  # nosec B113

    recengine = engine.SimilarityAnimeRecommendationEngine(mal.MyAnimeListProvider())

    recommendations = recengine.recommend_seasonal_anime_for_user(user, year, season)

    assert recommendations["title"].tolist() == [
        "Shingeki no Kyojin: The Final Season",
        "Golden Kamuy 4th Season",
    ]
    assert user_adapter.called
    assert season_adapter.called


def test_recommend_seasonal_anime_for_user_by_cluster(requests_mock, mocker):
    user = "Janiskeisari"

    url1 = f"{mal.MAL_API_URL}/users/{user}/animelist"
    user_adapter = requests_mock.get(url1, json=test_data.MAL_USER_LIST)  # nosec B113

    year = "2023"
    season = "winter"
    url2 = f"{mal.MAL_API_URL}/anime/season/{year}/{season}"
    season_adapter = requests_mock.get(url2, json=test_data.MAL_SEASONAL_LIST)  # nosec B113

    recengine = engine.ClusteringAnimeRecommendationEngine(mal.MyAnimeListProvider(), 2)
    recengine.NCLUSTERS = 2
    recommendations = recengine.recommend_seasonal_anime_for_user(user, year, season)

    assert recommendations["title"].tolist() == [
        "Golden Kamuy 4th Season",
        "Shingeki no Kyojin: The Final Season",
    ]
    assert user_adapter.called
    assert season_adapter.called
