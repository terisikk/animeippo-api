from animeippo.clustering import model

import animeippo.providers as providers

from animeippo import cache
from animeippo.recommendation.recommender import AnimeRecommender
from animeippo.recommendation import (
    engine,
    scoring,
    categories,
    encoding,
)


def get_default_scorers():
    return [
        scoring.FeatureCorrelationScorer(),
        ## scoring.FeatureSimilarityScorer(weighted=True),
        scoring.GenreAverageScorer(),
        scoring.ClusterSimilarityScorer(weighted=True),
        ## scoring.StudioCountScorer(),
        scoring.StudioCorrelationScorer(),
        scoring.PopularityScorer(),
        scoring.ContinuationScorer(),
        scoring.AdaptationScorer(),
        # scoring.SourceScorer(),
        scoring.DirectSimilarityScorer(),
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
        self._model = None
        self._seasonal_filters = None
        self._fetch_related_anime = False

    def build(self):
        return AnimeRecommender(self._provider, self._model, self._fetch_related_anime)

    def provider(self, provider):
        self._provider = provider
        return self

    def model(self, model):
        self._model = model
        return self

    def fetch_related_anime(self, fetch_related_anime):
        self._fetch_related_anime = fetch_related_anime
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
                        get_default_scorers(),
                        get_default_categorizers(metric),
                        model.AnimeClustering(
                            distance_metric=metric, distance_threshold=0.65, linkage="average"
                        ),
                        encoding.WeightedCategoricalEncoder(),
                    )
                )
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
                .fetch_related_anime(True)
            )
        case _:
            metric = "cosine"
            return (
                RecommenderBuilder()
                .provider(providers.mixed.MixedProvider(rcache))
                .model(
                    engine.AnimeRecommendationEngine(
                        get_default_scorers(),
                        get_default_categorizers(metric),
                        model.AnimeClustering(
                            distance_metric=metric, distance_threshold=0.65, linkage="average"
                        ),
                    )
                )
            )
