class AnimeRecommendationEngine:
    def __init__(self, scorers=None):
        self.scorers = scorers or []

    def fit_predict(self, user_dataset):
        recommendations = self.score_anime(user_dataset.seasonal, user_dataset.watchlist)

        return recommendations.sort_values("recommend_score", ascending=False)

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

    def add_scorer(self, scorer):
        self.scorers.append(scorer)
