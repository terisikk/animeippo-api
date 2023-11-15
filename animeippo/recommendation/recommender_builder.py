import asyncio

import animeippo.providers as providers
from animeippo import cache
from animeippo.recommendation.recommender import AnimeRecommender
from animeippo.recommendation import (
    engine,
    filters,
    scoring,
    dataset,
    categories,
    clustering,
    encoding,
)

import pandas as pd
import numpy as np


def get_default_scorers(distance_metric="jaccard"):
    return [
        # scoring.FeatureCorrelationScorer()
        ## scoring.FeatureSimilarityScorer(weighted=True),
        # scoring.GenreAverageScorer(),
        scoring.ClusterSimilarityScorer(weighted=True, distance_metric=distance_metric),
        ## scoring.StudioCountScorer(),
        # scoring.StudioAverageScorer(),
        # scoring.PopularityScorer(),
        # scoring.ContinuationScorer(),
        # scoring.SourceScorer(),
        # scoring.DirectSimilarityScorer(distance_metric=distance_metric),
    ]


def get_default_categorizers(distance_metric="jaccard"):
    return [
        categories.MostPopularCategory(),
        categories.SimulcastsCategory(),
        categories.ContinueWatchingCategory(),
        categories.YourTopPicksCategory(),
        categories.TopUpcomingCategory(),
        categories.ClusterCategory(0),
        categories.SourceCategory(),
        categories.ClusterCategory(1),
        categories.StudioCategory(),
        categories.ClusterCategory(2),
        categories.BecauseYouLikedCategory(0, distance_metric),
        categories.ClusterCategory(3),
        categories.BecauseYouLikedCategory(1, distance_metric),
        categories.ClusterCategory(4),
    ]


async def get_dataset(provider, user, year, season):
    if season is None:
        user_data, season_data1, season_data2, season_data3, season_data4 = await asyncio.gather(
            provider.get_user_anime_list(user),
            provider.get_seasonal_anime_list(year, "winter"),
            provider.get_seasonal_anime_list(year, "spring"),
            provider.get_seasonal_anime_list(year, "summer"),
            provider.get_seasonal_anime_list(year, "fall"),
        )

        season_data = pd.concat([season_data1, season_data2, season_data3, season_data4])
    else:
        user_data, season_data = await asyncio.gather(
            provider.get_user_anime_list(user),
            provider.get_seasonal_anime_list(year, season),
        )

    data = dataset.UserDataSet(
        user_data,
        season_data,
        provider.get_features(),
    )

    data.nsfw_tags += get_nswf_tags(user_data)
    data.nsfw_tags += get_nswf_tags(season_data)

    return data


def get_nswf_tags(df):
    if df is not None and "nsfw_tags" in df.columns:
        return df["nsfw_tags"].explode().dropna().unique().tolist()

    return []


def fill_user_status_data_from_watchlist(seasonal, watchlist):
    seasonal["user_status"] = np.nan
    seasonal["user_status"].update(watchlist["user_status"])
    return seasonal


async def construct_anilist_data(provider, year, season, user):
    data = await get_dataset(provider, user, year, season)

    if data.seasonal is not None and data.watchlist is not None:
        data.seasonal = fill_user_status_data_from_watchlist(data.seasonal, data.watchlist)
        data.seasonal = filters.ContinuationFilter(data.watchlist).filter(data.seasonal)

    if data.seasonal is not None:
        seasonal_filters = [
            filters.FeatureFilter("Kids", negative=True),
            filters.FeatureFilter("Hentai", negative=True),
            filters.StartSeasonFilter(
                (year, "winter"), (year, "spring"), (year, "summer"), (year, "fall")
            )
            if season is None
            else filters.StartSeasonFilter((year, season)),
        ]

        for f in seasonal_filters:
            data.seasonal = f.filter(data.seasonal)

    return data


async def get_related_anime(indices, provider):
    related_anime = []

    for i in indices:
        anime = await provider.get_related_anime(i)
        related_anime.append(anime.index.to_list())

    return related_anime


async def construct_myanimelist_data(provider, year, season, user):
    data = await get_dataset(provider, user, year, season)

    if data.seasonal is not None:
        seasonal_filters = [
            filters.MediaTypeFilter("tv"),
            filters.RatingFilter("g", "rx", negative=True),
            filters.StartSeasonFilter(
                (year, "winter"), (year, "spring"), (year, "summer"), (year, "fall")
            )
            if season is None
            else filters.StartSeasonFilter((year, season)),
        ]

        for f in seasonal_filters:
            data.seasonal = f.filter(data.seasonal)

        indices = data.seasonal.index.to_list()
        data.seasonal["relations"] = await get_related_anime(indices, provider)

    if data.watchlist is not None and data.seasonal is not None:
        data.seasonal = fill_user_status_data_from_watchlist(data.seasonal, data.watchlist)
        data.seasonal = filters.ContinuationFilter(data.watchlist).filter(data.seasonal)

    return data


class RecommenderBuilder:
    """Helps building a new anime recommender from several
    parts by returning self for each new part added, allowing
    chaining together different parts.

    Currently only uses one kind of recommender, so questionable
    if this class is really needed in between. In theory though
    this allows abstracting away different recommenders and
    also deferring building until build method is explicitly
    called.
    """

    def __init__(self):
        self._provider = None
        self._databuilder = None
        self._model = None
        self._seasonal_filters = None
        self._watchlist_filters = None

    def build(self):
        return AnimeRecommender(self._provider, self._model, self._databuilder)

    def provider(self, provider):
        self._provider = provider
        return self

    def databuilder(self, databuilder):
        self._databuilder = databuilder
        return self

    def model(self, model):
        self._model = model
        return self


def create_builder(providername):
    """
    Creates a recommender builder based on a third party data provider name.

    Different providers require a slightly different
    configuration to work effectively.

    Current options are "anilist" or "myanimelist".

    Final recommender is created when builder.build() is called.
    """
    rcache = cache.RedisCache()

    if not rcache.is_available():
        print("Warning: Redis cache is not available.")

    match providername:
        case "anilist":
            # Cosine seems to work better for anilist than jaccard.
            metric = "cosine"
            return (
                RecommenderBuilder()
                .provider(providers.anilist.AniListProvider(rcache))
                .model(
                    engine.AnimeRecommendationEngine(
                        get_default_scorers(metric),
                        get_default_categorizers(metric),
                        clustering.AnimeClustering(
                            distance_metric=metric, distance_threshold=0.65, linkage="average"
                        ),
                        encoding.WeightedCategoricalEncoder(),
                    )
                )
                .databuilder(construct_anilist_data)
            )
        case _:
            return (
                RecommenderBuilder()
                .provider(providers.myanimelist.MyAnimeListProvider(rcache))
                .model(
                    engine.AnimeRecommendationEngine(
                        get_default_scorers(), get_default_categorizers()
                    )
                )
                .databuilder(construct_myanimelist_data)
            )
