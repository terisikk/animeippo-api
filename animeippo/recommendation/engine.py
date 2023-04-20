import pandas as pd
from . import filters, analysis


class AnimeRecommendationEngine:
    def __init__(self, provider, filters=None, scorers=None):
        self.provider = provider
        self.scorers = scorers or []
        self.rec_filters = filters or []

    def recommend_seasonal_anime_for_user(self, user, year, season):
        seasonal_anime = self.provider.get_seasonal_anime_list(year, season)
        user_anime = self.provider.get_user_anime_list(user)

        analysis.fill_status_data_from_user_list(seasonal_anime, user_anime)

        user_anime_filtered = filters.IdFilter(*seasonal_anime, negative=True).filter(user_anime)

        seasonal_anime_filtered = pd.DataFrame(self.filter_anime(seasonal_anime))

        related_anime = []
        for i, row in seasonal_anime_filtered.iterrows():
            related_anime.append(self.provider.get_related_anime(i))

        seasonal_anime_filtered["related_anime"] = related_anime

        seasonal_anime_filtered = filters.ContinuationFilter(user_anime).filter(
            seasonal_anime_filtered
        )

        recommendations = self.score_anime(seasonal_anime_filtered, user_anime_filtered)

        return recommendations.sort_values("recommend_score", ascending=False).reset_index()

    def filter_anime(self, anime):
        filtered_df = anime

        for filter in self.rec_filters:
            filtered_df = filter.filter(filtered_df)

        return filtered_df

    def score_anime(self, scoring_target_df, compare_df):
        if len(self.scorers) > 0:
            names = []

            for scorer in self.scorers:
                scoring = scorer.score(scoring_target_df, compare_df)
                scoring_target_df.loc[:, scorer.name] = scoring
                names.append(scorer.name)

            scoring_target_df["recommend_score"] = scoring_target_df[names].mean(axis=1)
        else:
            raise RuntimeError("No scorers added for engine. Please add at least one.")

        return scoring_target_df

    def add_recommendation_filter(self, filter):
        self.rec_filters.append(filter)

    def add_scorer(self, scorer):
        self.scorers.append(scorer)
