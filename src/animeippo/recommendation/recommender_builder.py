from animeippo.profiling.model import UserProfile
from animeippo.recommendation.model import RecommendationModel

from .. import cache, providers
from ..analysis import encoding
from ..clustering import model
from . import categories, engine, scoring
from .recommender import AnimeRecommender


def get_default_scorers():
    """Get default scorers with optimized weights.

    Weights are tuned to balance different recommendation signals:
    - Direct similarity (0.25): Strongest signal for similar anime
    - Feature correlation (0.15): User's preference patterns
    - Cluster similarity (0.15): Similar viewing clusters
    - Continuation (0.15): Strong signal for sequels
    - Adaptation (0.10): Source material preferences
    - Popularity (0.10): Community consensus
    - Genre average (0.05): Basic genre preferences
    - Format (0.03): Format preferences (TV/Movie/etc)
    - Studio/Director (0.02 each): Production team correlation

    Total weight: 1.02 (intentionally slightly over 1.0 to boost strong signals)
    """
    return [
        scoring.DirectSimilarityScorer(weight=0.25),
        scoring.FeatureCorrelationScorer(weight=0.15),
        scoring.ClusterSimilarityScorer(weighted=True, weight=0.15),
        scoring.ContinuationScorer(weight=0.15),
        scoring.AdaptationScorer(weight=0.10),
        scoring.PopularityScorer(weight=0.10),
        scoring.GenreAverageScorer(weight=0.05),
        scoring.FormatScorer(weight=-0.3),  # Negative weight to penalize less preferred formats
        scoring.StudioCorrelationScorer(weight=0.02),
        scoring.DirectorCorrelationScorer(weight=0.02),
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


def build_recommender(providername):
    """
    Creates a recommender builder based on a third party data provider name.

    Different providers require a slightly different
    configuration to work effectively.

    Current options are "anilist" or "myanimelist".

    Final recommender is created when builder.build() is called.
    """
    default_recommendation_model_cls = RecommendationModel
    default_profile_model_cls = UserProfile

    rcache = cache.RedisCache()

    if not rcache.is_available():
        print("Warning: Redis cache is not available.")

    match providername:
        case "anilist":
            # Cosine seems to work better for anilist than jaccard.
            metric = "cosine"

            return AnimeRecommender(
                provider=providers.anilist.AniListProvider(rcache),
                engine=engine.AnimeRecommendationEngine(
                    model.AnimeClustering(
                        distance_metric=metric, distance_threshold=0.78, linkage="complete"
                    ),
                    encoding.WeightedCategoricalEncoder(),
                    get_default_scorers(),
                    get_default_categorizers(metric),
                ),
                recommendation_model_cls=default_recommendation_model_cls,
                profile_model_cls=default_profile_model_cls,
                fetch_related_anime=False,
            )

        case "myanimelist":
            return AnimeRecommender(
                provider=providers.myanimelist.MyAnimeListProvider(rcache),
                engine=engine.AnimeRecommendationEngine(
                    model.AnimeClustering(),
                    encoding.CategoricalEncoder(),
                    get_default_scorers(),
                    get_default_categorizers(),
                ),
                recommendation_model_cls=default_recommendation_model_cls,
                profile_model_cls=default_profile_model_cls,
                fetch_related_anime=True,
            )

        case _:
            metric = "cosine"
            return AnimeRecommender(
                provider=providers.mixed.MixedProvider(rcache),
                engine=engine.AnimeRecommendationEngine(
                    model.AnimeClustering(
                        distance_metric=metric, distance_threshold=0.65, linkage="average"
                    ),
                    encoding.WeightedCategoricalEncoder(),
                    get_default_scorers(),
                    get_default_categorizers(metric),
                ),
                recommendation_model_cls=default_recommendation_model_cls,
                profile_model_cls=default_profile_model_cls,
                fetch_related_anime=False,
            )
