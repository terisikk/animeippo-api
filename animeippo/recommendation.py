import animeippo.providers as providers
import animeippo.analysis as analysis


def recommend_seasonal_anime_for_mal_user(user, year, season):
    seasonal_anime = providers.myanimelist.get_seasonal_anime(year, season)
    user_anime = providers.myanimelist.get_user_anime(user)

    recommendations = analysis.order_by_recommendation(seasonal_anime, user_anime)

    return recommendations[recommendations["media_type"] == "tv"]
