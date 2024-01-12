import asyncio
from concurrent.futures import ThreadPoolExecutor


class AnimeRecommender:
    """Recommends new anime to a user if provided,
    or returns a filtered list of seasonal anime
    not tailored to a specific user."""

    def __init__(self, provider, engine, databuilder):
        self.provider = provider
        self.engine = engine
        self.databuilder = databuilder
        self.dataset = None

    def get_dataset(self, year, season, user=None):
        return self.databuilder(self.provider, year, season, user)

    def async_get_dataset(self, year, season, user=None):
        # If we run from jupyter, loop is already running and we need
        # to act differently. If the loop is not running,
        # we break into "normal path" with RuntimeError
        try:
            asyncio.get_running_loop()

            with ThreadPoolExecutor(1) as pool:
                print("using threading")

                return pool.submit(
                    lambda: asyncio.run(self.databuilder(self.provider, year, season, user))
                ).result()
        except RuntimeError:
            return asyncio.run(self.databuilder(self.provider, year, season, user))

    def recommend_seasonal_anime(self, year, season, user=None):
        recommendations = None

        dataset = self.async_get_dataset(year, season, user)

        if user:
            recommendations = self.engine.fit_predict(dataset)
        else:
            recommendations = dataset.seasonal.sort("popularity", descending=True)

        dataset.recommendations = recommendations

        return dataset

    def get_categories(self, recommendations):
        return self.engine.categorize_anime(recommendations)
