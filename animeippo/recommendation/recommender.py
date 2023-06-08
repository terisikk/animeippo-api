import asyncio


class AnimeRecommender:
    def __init__(self, provider, engine, databuilder):
        self.provider = provider
        self.engine = engine
        self.databuilder = databuilder
        self.dataset = None

    def get_dataset(self, year, season, user=None):
        return self.databuilder(self.provider, year, season, user)

    def async_get_dataset(self, year, season, user=None):
        return asyncio.run(self.databuilder(self.provider, year, season, user))

    def recommend_seasonal_anime(self, year, season, user=None):
        recommendations = None

        self.dataset = self.async_get_dataset(year, season, user)

        if user:
            recommendations = self.engine.fit_predict(self.dataset)
        else:
            recommendations = self.dataset.seasonal.sort_values("popularity", ascending=False)

        self.dataset.recommendations = recommendations

        return self.dataset

    def get_categories(self, recommendations):
        return self.engine.categorize_anime(recommendations)
