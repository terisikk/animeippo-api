import animeippo.providers as providers
import animeippo.analysis as analysis


def recommend_seasonal_anime_for_mal_user(user, year, season, weighted=True):
    provider = providers.myanimelist.MyAnimeListProvider()

    seasonal_anime = provider.get_seasonal_anime_list(year, season)
    user_anime = provider.get_user_anime_list(user)

    recommendations = analysis.recommend_by_genre_similarity(seasonal_anime, user_anime, weighted)

    return recommendations[recommendations["media_type"] == "tv"]
