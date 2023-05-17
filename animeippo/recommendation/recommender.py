class AnimeRecommender:
    def __init__(self, provider, engine, databuilder):
        self.provider = provider
        self.engine = engine
        self.databuilder = databuilder

    def recommend_seasonal_anime(self, year, season, user=None):
        recommendations = None

        dataset = self.databuilder(year, season, user)

        if user:
            recommendations = self.engine.fit_predict(dataset)
        else:
            recommendations = dataset.seasonal.sort_values("popularity", ascending=False)

        return recommendations
