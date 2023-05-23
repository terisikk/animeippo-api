import abc
import asyncio

import animeippo.providers as providers
from animeippo import cache
from animeippo.recommendation.recommender import AnimeRecommender
from animeippo.recommendation import engine, filters, scoring, dataset


DEFAULT_SCORERS = [
    # scoring.FeatureSimilarityScorer(feature_tags, weighted=True),
    scoring.GenreAverageScorer(),
    scoring.ClusterSimilarityScorer(weighted=True),
    # scoring.StudioCountScorer(),
    scoring.StudioAverageScorer(),
    scoring.PopularityScorer(),
    scoring.ContinuationScorer(),
    scoring.SourceScorer(),
    scoring.DirectSimilarityScorer(),
]


class AbstractRecommenderBuilder(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def build(self):
        pass

    @abc.abstractmethod
    def _build_provider(self):
        pass

    @abc.abstractmethod
    def _build_databuilder(self, provider, user, year, season):
        pass

    @abc.abstractmethod
    def _build_model(self):
        pass


class AniListRecommenderBuilder(AbstractRecommenderBuilder):
    def build(self):
        provider = self._build_provider()
        databuilder = self._build_databuilder(provider)
        model = self._build_model()

        return AnimeRecommender(provider, model, databuilder)

    def _build_provider(self):
        rcache = cache.RedisCache()
        return providers.anilist.AniListProvider(cache=rcache)

    def _build_databuilder(self, provider):
        async def databuilder(year, season, user):
            user_data, season_data = await asyncio.gather(
                provider.get_user_anime_list(user),
                provider.get_seasonal_anime_list(year, season),
            )

            data = dataset.UserDataSet(
                user_data,
                season_data,
                provider.get_features(),
            )

            if data.seasonal is not None and data.watchlist is not None:
                watchlist_filters = [
                    filters.IdFilter(*data.seasonal.index.to_list(), negative=True)
                ]

                for f in watchlist_filters:
                    data.watchlist = f.filter(data.watchlist)

                data.seasonal = filters.ContinuationFilter(data.watchlist).filter(data.seasonal)

            if data.seasonal is not None:
                seasonal_filters = [
                    filters.FeatureFilter("Kids", negative=True),
                    filters.FeatureFilter("Hentai", negative=True),
                    filters.StartSeasonFilter((year, season)),
                ]

                for f in seasonal_filters:
                    data.seasonal = f.filter(data.seasonal)

            return data

        return databuilder

    def _build_model(self):
        return engine.AnimeRecommendationEngine(DEFAULT_SCORERS)


class MyAnimeListRecommenderBuilder(AbstractRecommenderBuilder):
    def build(self):
        provider = self._build_provider()
        dataset = self._build_databuilder(provider)
        model = self._build_model()

        return AnimeRecommender(provider, model, dataset)

    def _build_provider(self):
        rcache = cache.RedisCache()
        return providers.myanimelist.MyAnimeListProvider(cache=rcache)

    def _build_databuilder(self, provider):
        async def get_related_anime(indices, provider):
            related_anime = []

            for i in indices:
                anime = await provider.get_related_anime(i)
                related_anime.append(anime.index.to_list())

            return related_anime

        async def databuilder(year, season, user):
            user_data, season_data = await asyncio.gather(
                provider.get_user_anime_list(user),
                provider.get_seasonal_anime_list(year, season),
            )

            data = dataset.UserDataSet(
                user_data,
                season_data,
                provider.get_features(),
            )

            if data.seasonal is not None:
                seasonal_filters = [
                    filters.MediaTypeFilter("tv"),
                    filters.RatingFilter("g", "rx", negative=True),
                    filters.StartSeasonFilter((year, season)),
                ]

                for f in seasonal_filters:
                    data.seasonal = f.filter(data.seasonal)

                indices = data.seasonal.index.to_list()
                data.seasonal["related_anime"] = await get_related_anime(indices, provider)

            if data.watchlist is not None and data.seasonal is not None:
                watchlist_filters = [
                    filters.IdFilter(*data.seasonal.index.to_list(), negative=True)
                ]

                for f in watchlist_filters:
                    data.watchlist = f.filter(data.watchlist)

                data.seasonal = filters.ContinuationFilter(data.watchlist).filter(data.seasonal)

            return data

        return databuilder

    def _build_model(self):
        return engine.AnimeRecommendationEngine(DEFAULT_SCORERS)


def create_builder(providername):
    match providername:
        case "anilist":
            return AniListRecommenderBuilder()
        case _:
            return MyAnimeListRecommenderBuilder()
