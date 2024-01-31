from animeippo.profiling.model import UserProfile
from animeippo.recommendation.model import RecommendationModel

from .. import cache, providers
from ..analysis import encoding
from ..clustering import model
from . import categories, engine, scoring
from .recommender import AnimeRecommender


def get_default_scorers():
    return [
        scoring.FeatureCorrelationScorer(),
        scoring.GenreAverageScorer(),
        scoring.ClusterSimilarityScorer(weighted=True),
        scoring.StudioCorrelationScorer(),
        scoring.PopularityScorer(),
        scoring.ContinuationScorer(),
        scoring.AdaptationScorer(),
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
        categories.DiscouragingWrapper(categories.GenreCategory(0)),
        categories.AdaptationCategory(),
        categories.DiscouragingWrapper(categories.GenreCategory(1)),
        categories.PlanningCategory(),
        categories.DiscouragingWrapper(categories.GenreCategory(2)),
        categories.SourceCategory(),
        categories.DiscouragingWrapper(categories.GenreCategory(3)),
        categories.StudioCategory(),
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
        self._engine = engine.AnimeRecommendationEngine(
            get_default_scorers(),
            get_default_categorizers(),
            model.AnimeClustering(),
            encoding.CategoricalEncoder(),
        )
        self._fetch_related_anime = False
        self._recommendation_model_cls = RecommendationModel
        self._profile_model_cls = UserProfile

    def build(self):
        return AnimeRecommender(
            provider=self._provider,
            engine=self._engine,
            recommendation_model_cls=self._recommendation_model_cls,
            profile_model_cls=self._profile_model_cls,
            fetch_related_anime=self._fetch_related_anime,
        )

    def provider(self, provider):
        self._provider = provider
        return self

    def engine(self, engine):
        self._engine = engine
        return self

    def fetch_related_anime(self, fetch_related_anime):
        self._fetch_related_anime = fetch_related_anime
        return self

    def recommendation_model_class(self, cls):
        self._recommendation_model_cls = cls

        return self

    def profile_model_class(self, cls):
        self._profile_model_cls = cls
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
                .engine(
                    engine.AnimeRecommendationEngine(
                        model.AnimeClustering(
                            distance_metric="cosine", distance_threshold=0.78, linkage="complete"
                        ),
                        encoding.WeightedCategoricalEncoder(),
                        get_default_scorers(),
                        get_default_categorizers(metric),
                    )
                )
            )
        case "myanimelist":
            return (
                RecommenderBuilder()
                .provider(providers.myanimelist.MyAnimeListProvider(rcache))
                .engine(
                    engine.AnimeRecommendationEngine(
                        model.AnimeClustering(),
                        encoding.CategoricalEncoder(),
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
                .engine(
                    engine.AnimeRecommendationEngine(
                        model.AnimeClustering(
                            distance_metric=metric, distance_threshold=0.65, linkage="average"
                        ),
                        encoding.WeightedCategoricalEncoder(),
                        get_default_scorers(),
                        get_default_categorizers(metric),
                    )
                )
            )
