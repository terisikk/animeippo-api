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


def get_default_categorizers(distance_metric="cosine", tag_lookup=None, genres=None):
    """Get categorizer layouts for different data volumes.

    Returns dict of layouts. The orchestrator selects at render time based
    on recommendation count: minimal (<20), standard (20-100), full (>100).
    Each category also has min_items — skipped if too few items match.
    """
    debug = (
        [(categories.DebugCategory(), None)]
        if os.getenv("DEBUG", "false").lower() == "true"
        else []
    )

    minimal = [
        *debug,
        (categories.TopUpcomingCategory(min_items=3), 35),
        (categories.ContinueWatchingCategory(), None),
        (categories.YourTopPicksCategory(), 10),
        (categories.MostPopularCategory(), 20),
    ]

    standard = [
        *debug,
        (categories.TopReleasedPicksCategory(), 3),
        (categories.ContinueWatchingCategory(), None),
        (categories.SimulcastsCategory(min_items=3), 40),
        (categories.TopUpcomingCategory(min_items=3), 35),
        (categories.PlanningCategory(), None),
        (categories.MovieNightCategory(), 3),
        (categories.YourTopPicksCategory(), 35),
        (categories.MostPopularCategory(), 20),
        (categories.HiddenGemsCategory(), 3),
        (categories.StudioCategory(min_items=2), None),
        (categories.AllMoviesCategory(), None),
        (categories.AdaptationCategory(), None),
    ]

    cluster_kwargs = {"tag_lookup": tag_lookup or {}, "genres": genres or set(), "min_items": 3}

    full = [
        *debug,
        (categories.TopReleasedPicksCategory(), 3),
        (categories.ContinueWatchingCategory(), None),
        (categories.SimulcastsCategory(min_items=3), 40),
        (categories.TopUpcomingCategory(min_items=3), 35),
        (categories.PlanningCategory(), None),
        (categories.MovieNightCategory(), 3),
        (categories.YourTopPicksCategory(), 35),
        (categories.ClusterCategory(nth_cluster=0, **cluster_kwargs), 20),
        (categories.BecauseYouLikedCategory(nth_liked=0, distance_metric=distance_metric), 20),
        (categories.MostPopularCategory(), 20),
        (categories.HiddenGemsCategory(), 3),
        (categories.ClusterCategory(nth_cluster=1, **cluster_kwargs), 20),
        (categories.GenreCategory(nth_genre=0, needs_diversity=True, min_items=2), None),
        (categories.BecauseYouLikedCategory(nth_liked=1, distance_metric=distance_metric), 20),
        (categories.AdaptationCategory(), None),
        (categories.ClusterCategory(nth_cluster=2, **cluster_kwargs), 20),
        (categories.GenreCategory(nth_genre=1, needs_diversity=True, min_items=2), None),
        (categories.StudioCategory(min_items=2), None),
        (categories.ClusterCategory(nth_cluster=3, **cluster_kwargs), 20),
        (categories.GenreCategory(nth_genre=2, needs_diversity=True, min_items=2), None),
        (categories.MangaCategory(), 25),
        (categories.BecauseYouLikedCategory(nth_liked=2, distance_metric=distance_metric), 20),
        (categories.ClusterCategory(nth_cluster=4, **cluster_kwargs), 20),
        (categories.GenreCategory(nth_genre=3, needs_diversity=True, min_items=2), None),
        (categories.AllMoviesCategory(), None),
    ]

    return {"minimal": minimal, "standard": standard, "full": full}


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
            ranking_orchestrator=RankingOrchestrator(
                get_default_categorizers(
                    distance_metric=metric,
                    tag_lookup=provider.get_tag_lookup(),
                    genres=provider.get_genres(),
                )
            ),
        ),
        recommendation_model_cls=RecommendationModel,
        profile_model_cls=UserProfile,
        fetch_related_anime=False,
    )
