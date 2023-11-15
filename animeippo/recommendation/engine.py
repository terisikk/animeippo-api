import numpy as np
import pandas as pd


from animeippo.recommendation import encoding, clustering


class AnimeRecommendationEngine:
    """Generates recommendations based on given scorers and categorizers. Optionally accepts
    a custom clustering model and feature encoder."""

    def __init__(self, scorers=None, categorizers=None, clustering_model=None, encoder=None):
        self.scorers = scorers or []
        self.categorizers = categorizers or []
        self.clustering_model = clustering_model or clustering.AnimeClustering()
        self.encoder = encoder or encoding.CategoricalEncoder()

    def validate(self, dataset):
        is_missing_seasonal = dataset.seasonal is None
        is_missing_watchlist = dataset.watchlist is None

        if is_missing_seasonal or is_missing_watchlist:
            error_desc = (
                f"Watchlist invalid?: {is_missing_watchlist}. "
                + f"Seasonal invalid?: {is_missing_seasonal}"
            )

            raise RuntimeError("Trying to recommend anime without proper data. " + error_desc)

    def fit(self, dataset):
        self.validate(dataset)

        dataset.all_features = (
            pd.concat(
                [
                    dataset.all_features if dataset.all_features is not None else pd.Series(),
                    dataset.seasonal["features"],
                    dataset.watchlist["features"],
                ]
            )
            .explode()
            .dropna()
            .unique()
        )

        self.encoder.fit(dataset.all_features)

        dataset.watchlist["encoded"] = self.encoder.encode(dataset.watchlist)
        dataset.seasonal["encoded"] = self.encoder.encode(dataset.seasonal)

        dataset.watchlist["cluster"] = self.get_clustering(dataset.watchlist["encoded"])

        return dataset

    def fit_predict(self, dataset):
        dataset = self.fit(dataset)

        recommendations = self.score_anime(dataset)

        recommendations["cluster"] = self.clustering_model.predict(dataset.seasonal["encoded"])

        return recommendations.sort_values("recommend_score", ascending=False)

    def score_anime(self, dataset):
        scoring_target_df = dataset.seasonal

        if len(self.scorers) > 0:
            names = []

            for scorer in self.scorers:
                scoring = scorer.score(dataset)
                scoring_target_df.loc[:, scorer.name] = scoring
                names.append(scorer.name)

            scoring_target_df["recommend_score"] = scoring_target_df[names].mean(axis=1)
        else:
            raise RuntimeError("No scorers added for engine. Please add at least one.")

        return scoring_target_df

    def categorize_anime(self, data):
        cats = []

        for grouper in self.categorizers:
            group = grouper.categorize(data)

            if group is not None and len(group) > 0:
                items = group.index.tolist()
                cats.append({"name": grouper.description, "items": items})

        return cats

    def add_scorer(self, scorer):
        self.scorers.append(scorer)

    def add_categorizer(self, categorizer):
        self.categorizers.append(categorizer)

    def get_clustering(self, series):
        encoded = np.vstack(series)
        return self.clustering_model.cluster_by_features(encoded, series.index)
