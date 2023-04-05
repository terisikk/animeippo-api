import animeippo.recommendation.engine as engine
import animeippo.recommendation.scoring as scoring
import animeippo.providers.myanimelist as mal
import animeippo.recommendation.filters as filters

from tests import test_data


# Figure out how to provide correct data from this, mal data is not formatted
class ProviderStub:
    def get_seasonal_anime_list(self, *args, **kwargs):
        return test_data.MAL_SEASONAL_LIST

    def get_user_anime_list(self, *args, **kwargs):
        return test_data.MAL_USER_LIST


def test_recommend_seasonal_anime_for_user_by_genre(requests_mock):
    user = "Janiskeisari"

    url1 = f"{mal.MAL_API_URL}/users/{user}/animelist"
    user_adapter = requests_mock.get(url1, json=test_data.MAL_USER_LIST)  # nosec B113

    year = "2023"
    season = "winter"
    url2 = f"{mal.MAL_API_URL}/anime/season/{year}/{season}"
    season_adapter = requests_mock.get(url2, json=test_data.MAL_SEASONAL_LIST)  # nosec B113

    encoder = engine.CategoricalEncoder(mal.MAL_GENRES)
    scorer = scoring.GenreSimilarityScorer()
    recengine = engine.AnimeRecommendationEngine(mal.MyAnimeListProvider(), scorer, encoder)

    recommendations = recengine.recommend_seasonal_anime_for_user(user, year, season)

    assert recommendations["title"].tolist() == [
        "Shingeki no Kyojin: The Final Season",
        "Golden Kamuy 4th Season",
    ]
    assert user_adapter.called
    assert season_adapter.called


def test_recommend_seasonal_anime_for_user_by_cluster(requests_mock):
    user = "Janiskeisari"

    url1 = f"{mal.MAL_API_URL}/users/{user}/animelist"
    user_adapter = requests_mock.get(url1, json=test_data.MAL_USER_LIST)  # nosec B113

    year = "2023"
    season = "winter"
    url2 = f"{mal.MAL_API_URL}/anime/season/{year}/{season}"
    season_adapter = requests_mock.get(url2, json=test_data.MAL_SEASONAL_LIST)  # nosec B113

    scorer = scoring.ClusterSimilarityScorer(2)
    encoder = engine.CategoricalEncoder(mal.MAL_GENRES)

    recengine = engine.AnimeRecommendationEngine(mal.MyAnimeListProvider(), scorer, encoder)
    recommendations = recengine.recommend_seasonal_anime_for_user(user, year, season)

    assert recommendations["title"].tolist() == [
        "Golden Kamuy 4th Season",
        "Shingeki no Kyojin: The Final Season",
    ]
    assert user_adapter.called
    assert season_adapter.called


def test_filters_work(requests_mock):
    user = "Janiskeisari"

    url1 = f"{mal.MAL_API_URL}/users/{user}/animelist"
    user_adapter = requests_mock.get(url1, json=test_data.MAL_USER_LIST)  # nosec B113

    year = "2023"
    season = "winter"
    url2 = f"{mal.MAL_API_URL}/anime/season/{year}/{season}"
    season_adapter = requests_mock.get(url2, json=test_data.MAL_SEASONAL_LIST)  # nosec B113

    encoder = engine.CategoricalEncoder(mal.MAL_GENRES)
    scorer = scoring.ClusterSimilarityScorer(2)

    recengine = engine.AnimeRecommendationEngine(
        mal.MyAnimeListProvider(),
        scorer,
        encoder,
    )

    recengine.add_recommendation_filter(filters.GenreFilter("Gore", negative=True))

    recommendations = recengine.recommend_seasonal_anime_for_user(user, year, season)

    assert len(recommendations.index) == 1
