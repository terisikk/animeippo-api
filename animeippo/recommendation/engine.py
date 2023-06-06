import numpy as np
import pandas as pd

from animeippo.recommendation import encoding


class AnimeRecommendationEngine:
    def __init__(self, scorers=None):
        self.scorers = scorers or []

    def validate(self, dataset):
        is_missing_seasonal = dataset.seasonal is None
        is_missing_watchlist = dataset.watchlist is None

        if is_missing_seasonal or is_missing_watchlist:
            error_desc = (
                f"Watchlist invalid?: {is_missing_watchlist}."
                + "Seasonal invalid?: {is_missing_seasonal}"
            )

            raise RuntimeError("Trying to recommend anime without proper data. " + error_desc)

    def fit(self, dataset):
        self.validate(dataset)

        dataset.seasonal = self.fill_status_data_from_watchlist(dataset.seasonal, dataset.watchlist)

        dataset.all_features = pd.concat(
            [dataset.all_features, dataset.seasonal["features"], dataset.watchlist["features"]]
        )

        encoder = encoding.CategoricalEncoder(dataset.all_features.explode().unique())

        dataset.watchlist["encoded"] = encoder.encode(dataset.watchlist["features"]).tolist()
        dataset.seasonal["encoded"] = encoder.encode(dataset.seasonal["features"]).tolist()

        return dataset

    def fit_predict(self, dataset):
        dataset = self.fit(dataset)

        recommendations = self.score_anime(dataset.seasonal, dataset.watchlist)

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

    def fill_status_data_from_watchlist(self, seasonal, watchlist):
        seasonal["status"] = np.nan
        seasonal["status"].update(watchlist["status"])
        return seasonal
