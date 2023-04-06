from . import filters


class AnimeRecommendationEngine:
    def __init__(self, provider):
        self.provider = provider
        self.scorers = []
        self.rec_filters = []

    def recommend_seasonal_anime_for_user(self, user, year, season):
        seasonal_anime = self.provider.get_seasonal_anime_list(year, season)
        user_anime = self.provider.get_user_anime_list(user)

        user_anime_filtered = filters.IdFilter(*seasonal_anime["id"], negative=True).filter(
            user_anime
        )

        seasonal_anime_filtered = self.filter_anime(seasonal_anime)

        recommendations = self.score_anime(seasonal_anime_filtered, user_anime_filtered)

        return recommendations.sort_values("recommend_score", ascending=False)

    def filter_anime(self, anime):
        filtered_df = anime

        for filter in self.rec_filters:
            filtered_df = filter.filter(filtered_df)

        return filtered_df.reset_index(drop=True)

    def score_anime(self, scoring_target_df, compare_df):
        if len(self.scorers) > 0:
            scoring_target_df["recommend_score"] = 0

            for scorer in self.scorers:
                scoring_target_df["recommend_score"] = (
                    scoring_target_df["recommend_score"]
                    + scorer.score(scoring_target_df, compare_df)[0]
                )

            scoring_target_df["recommend_score"] = scoring_target_df["recommend_score"] / len(
                self.scorers
            )
        else:
            raise RuntimeError("No scorers added for engine. Please add at least one.")

        return scoring_target_df

    def add_recommendation_filter(self, filter):
        self.rec_filters.append(filter)

    def add_scorer(self, scorer):
        self.scorers.append(scorer)
