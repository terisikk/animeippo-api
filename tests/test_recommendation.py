import animeippo.recommendation.engine as engine
import animeippo.providers.myanimelist as mal

from tests import test_data


def test_recommend_seasonal_anime_for_mal_user(requests_mock):
    user = "Janiskeisari"

    url1 = f"{mal.MAL_API_URL}/users/{user}/animelist"
    user_adapter = requests_mock.get(url1, json=test_data.MAL_DATA)  # nosec B113

    year = "2023"
    season = "winter"
    url2 = f"{mal.MAL_API_URL}/anime/season/{year}/{season}"
    season_adapter = requests_mock.get(url2, json=test_data.MAL_DATA)  # nosec B113

    recommendations = engine.recommend_seasonal_anime_for_mal_user(user, year, season)

    assert recommendations["title"].tolist() == ["Neon Genesis Evangelion", "Hellsing"]
    assert user_adapter.called
    assert season_adapter.called
