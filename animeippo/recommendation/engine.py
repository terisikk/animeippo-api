from . import filters


class AnimeRecommendationEngine:
    provider = None
    scorer = None
    rec_filters = []

    def __init__(self, provider, scorer):
        self.provider = provider
        self.scorer = scorer

    def recommend_seasonal_anime_for_user(self, user, year, season):
        seasonal_anime = self.provider.get_seasonal_anime_list(year, season)
        user_anime = self.provider.get_user_anime_list(user)

        user_anime_filtered = filters.IdFilter(*seasonal_anime["id"], negative=True).filter(
            user_anime
        )

        seasonal_anime_filtered = self.filter_anime(seasonal_anime)

        recommendations = self.scorer.score(seasonal_anime_filtered, user_anime_filtered)

        return recommendations

    def filter_anime(self, anime):
        filtered_df = anime

        for filter in self.rec_filters:
            filtered_df = filter.filter(filtered_df)

        return filtered_df

    def add_recommendation_filter(self, filter):
        self.rec_filters.append(filter)
