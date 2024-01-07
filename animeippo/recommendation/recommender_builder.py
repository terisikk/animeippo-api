import asyncio

from async_lru import alru_cache

import animeippo.providers as providers
import polars as pl

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
    profile,
)


def get_default_scorers(distance_metric="jaccard"):
    return [
        scoring.FeatureCorrelationScorer(),
        ## scoring.FeatureSimilarityScorer(weighted=True),
        scoring.GenreAverageScorer(),
        scoring.ClusterSimilarityScorer(weighted=True, distance_metric=distance_metric),
        ## scoring.StudioCountScorer(),
        scoring.StudioCorrelationScorer(),
        scoring.PopularityScorer(),
        scoring.ContinuationScorer(),
        scoring.AdaptationScorer(),
        # scoring.SourceScorer(),
        scoring.DirectSimilarityScorer(distance_metric=distance_metric),
        scoring.FormatScorer(),
        scoring.DirectorCorrelationScorer(),
    ]


def get_default_categorizers(distance_metric="jaccard"):
    return [
        categories.MostPopularCategory(),
        categories.SimulcastsCategory(),
        categories.ContinueWatchingCategory(),
        categories.YourTopPicksCategory(),
        # categories.DebugCategory(),
        categories.TopUpcomingCategory(),
        # categories.ClusterCategory(0),
        categories.DiscouragingWrapper(categories.GenreCategory(0)),
        categories.AdaptationCategory(),
        # categories.ClusterCategory(1),
        categories.DiscouragingWrapper(categories.GenreCategory(1)),
        categories.PlanningCategory(),
        # categories.ClusterCategory(2),
        categories.DiscouragingWrapper(categories.GenreCategory(2)),
        categories.SourceCategory(),
        # categories.ClusterCategory(3),
        categories.DiscouragingWrapper(categories.GenreCategory(3)),
        categories.StudioCategory(),
        # categories.ClusterCategory(4),
        categories.DiscouragingWrapper(categories.GenreCategory(4)),
        categories.BecauseYouLikedCategory(0, distance_metric),
        categories.DiscouragingWrapper(categories.GenreCategory(5)),
        categories.BecauseYouLikedCategory(1, distance_metric),
        categories.DiscouragingWrapper(categories.GenreCategory(6)),
        categories.BecauseYouLikedCategory(2, distance_metric),
    ]


@alru_cache(maxsize=1)
async def get_user_profile(provider, user):
    user_data = await provider.get_user_anime_list(user)
    user_profile = profile.UserProfile(user, user_data)

    return user_profile


async def get_dataset(provider, user, year, season):
    user_profile, manga_data, season_data = await asyncio.gather(
        get_user_profile(provider, user),
        provider.get_user_manga_list(user),
        provider.get_seasonal_anime_list(year, season),
    )

    user_profile.mangalist = manga_data

    data = dataset.RecommendationModel(user_profile, season_data)

    data.nsfw_tags += get_nswf_tags(user_profile.watchlist)
    data.nsfw_tags += get_nswf_tags(season_data)

    return data


def get_nswf_tags(df):
    if df is not None and "nsfw_tags" in df.columns:
        return df["nsfw_tags"].explode().unique().drop_nans().to_list()

    return []


def fill_user_status_data_from_watchlist(seasonal, watchlist):
    return seasonal.join(watchlist.select(["id", "user_status"]), on="id", how="left")


async def construct_anilist_data(provider, year, season, user):
    data = await get_dataset(provider, user, year, season)

    if data.seasonal is not None and data.watchlist is not None:
        data.seasonal = fill_user_status_data_from_watchlist(data.seasonal, data.watchlist)
        data.seasonal = data.seasonal.filter(filters.ContinuationFilter(data.watchlist))

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

        data.seasonal = data.seasonal.filter(seasonal_filters)

    return data


async def get_related_anime(indices, provider):
    related_anime = []

    for i in indices:
        anime = await provider.get_related_anime(i)
        related_anime.append(anime)

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

        data.seasonal = data.seasonal.filter(seasonal_filters)

        indices = data.seasonal["id"].to_list()
        related_anime = await get_related_anime(indices, provider)
        data.seasonal.with_columns(continuation_to=related_anime)

    if data.watchlist is not None and data.seasonal is not None:
        data.seasonal = fill_user_status_data_from_watchlist(data.seasonal, data.watchlist)
        data.seasonal = data.seasonal.filter(filters.ContinuationFilter(data.watchlist))

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
        case "myanimelist":
            return (
                RecommenderBuilder()
                .provider(providers.myanimelist.MyAnimeListProvider(rcache))
                .model(
                    engine.AnimeRecommendationEngine(
                        get_default_scorers(),
                        get_default_categorizers(),
                    )
                )
                .databuilder(construct_myanimelist_data)
            )
        case _:
            metric = "cosine"
            return (
                RecommenderBuilder()
                .provider(providers.mixed_provider.MixedProvider(rcache))
                .model(
                    engine.AnimeRecommendationEngine(
                        get_default_scorers(metric),
                        get_default_categorizers(metric),
                        clustering.AnimeClustering(
                            distance_metric=metric, distance_threshold=0.65, linkage="average"
                        ),
                    )
                )
                .databuilder(construct_anilist_data)
            )
