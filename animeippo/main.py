from animeippo.providers import myanimelist as mal
from animeippo.recommendation import engine, filters, scoring


def create_engine():
    provider = mal.MyAnimeListProvider()
    encoder = engine.CategoricalEncoder(mal.MAL_GENRES)
    scorer = scoring.ClusterSimilarityScorer()

    recommender = engine.AnimeRecommendationEngine(provider, scorer, encoder)

    recommender.add_recommendation_filter(filters.GenreFilter("Kids", negative=True))
    recommender.add_recommendation_filter(filters.MediaTypeFilter("tv"))

    return recommender


if __name__ == "__main__":
    recommender = create_engine()

    recommendations = recommender.recommend_seasonal_anime_for_user(
        "Janiskeisari", "2023", "winter"
    )
    print(recommendations[0:25])
