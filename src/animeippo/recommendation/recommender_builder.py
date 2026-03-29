import os

import structlog

from animeippo.profiling.model import UserProfile
from animeippo.recommendation.model import RecommendationModel

from .. import cache, providers
from ..analysis import encoding
from ..clustering import model
from . import categories, engine, scoring
from .ranking import RankingOrchestrator
from .recommender import AnimeRecommender

logger = structlog.get_logger()


def get_discovery_scorers():
    """Scorers for discovering new anime based on taste similarity."""
    return [
        # Capture shows that closely resemble previous shows the user liked
        scoring.DirectSimilarityScorer(weight=0.30),
        # Capture shows that share specific features with user preferences
        scoring.FeatureCorrelationScorer(weight=0.25),
        # Capture shows that are similar to user's clusters of interest
        scoring.ClusterSimilarityScorer(weight=0.20),
        # Capture shows from studios the user has liked before
        scoring.StudioCorrelationScorer(weight=0.10),
        # Capture generally popular shows to ensure relevance
        scoring.PopularityScorer(weight=0.10),
        # Capture shows that are adaptations of manga the user has read
        scoring.AdaptationScorer(weight=0.05),
    ]


def get_engagement_scorers():
    """Scorers based on prior user engagement (watched prequels)."""
    return [
        scoring.ContinuationScorer(weight=0.15),
    ]


def get_default_categorizers(distance_metric="jaccard"):
    """Get default categorizers with their top_n limits.

    Returns list of (category, top_n) tuples where:
    - top_n=None means no limit (diversity-adjusted categories)
    - top_n=N means return top N items
    """
    categorizer_list = [
        (categories.TopReleasedPicksCategory(), 3),
        (categories.ContinueWatchingCategory(), None),
        (categories.HiddenGemsCategory(), 3),
        (categories.MostPopularCategory(), 20),
        (categories.SimulcastsCategory(), 40),
        (categories.YourTopPicksCategory(), 35),
        (categories.TopUpcomingCategory(), 35),
        (categories.GenreCategory(0), None),
        (categories.AdaptationCategory(), None),
        (categories.GenreCategory(1), None),
        (categories.TopMoviesCategory(), 3),
        (categories.PlanningCategory(), None),
        (categories.GenreCategory(2), None),
        (categories.MangaCategory(), 25),
        (categories.GenreCategory(3), None),
        (categories.StudioCategory(), 25),
        (categories.GenreCategory(4), None),
        (categories.BecauseYouLikedCategory(0, distance_metric), 20),
        (categories.GenreCategory(5), None),
        (categories.BecauseYouLikedCategory(1, distance_metric), 20),
        (categories.GenreCategory(6), None),
        (categories.BecauseYouLikedCategory(2, distance_metric), 20),
    ]

    if os.getenv("DEBUG", "false").lower() == "true":
        categorizer_list.insert(0, (categories.DebugCategory(), None))

    return categorizer_list


def build_recommender(providername):
    """
    Creates a recommender builder based on a third party data provider name.

    Different providers require a slightly different
    configuration to work effectively.

    Current options are "anilist" or "myanimelist".

    Final recommender is created when builder.build() is called.
    """
    rcache = cache.RedisCache()

    if not rcache.is_available():
        logger.warning("redis_unavailable")

    match providername:
        case "anilist":
            provider = providers.anilist.AniListProvider(rcache)
        case _:
            provider = providers.mixed.MixedProvider(rcache)

    metric = "cosine"

    return AnimeRecommender(
        provider=provider,
        engine=engine.AnimeRecommendationEngine(
            model.AnimeClustering(
                distance_metric=metric,
                distance_threshold=0.63,
                linkage="average",
                min_cluster_size=3,
                franchise_reduction=True,
            ),
            encoding.WeightedCategoricalEncoder(),
            discovery_scorers=get_discovery_scorers(),
            engagement_scorers=get_engagement_scorers(),
            ranking_orchestrator=RankingOrchestrator(get_default_categorizers(metric)),
        ),
        recommendation_model_cls=RecommendationModel,
        profile_model_cls=UserProfile,
        fetch_related_anime=False,
    )
