from animeippo.providers import myanimelist as mal
from animeippo.recommendation import engine, filters, scoring


def create_recommender():
    provider = mal.MyAnimeListProvider()
    encoder = scoring.CategoricalEncoder(provider.get_genre_tags())
    scorer = scoring.ClusterSimilarityScorer(encoder, weighted=True)

    # scorer = scoring.StudioSimilarityScorer(weighted=False)

    recommender = engine.AnimeRecommendationEngine(provider)

    recommender.add_scorer(scorer)

    recommender.add_recommendation_filter(filters.GenreFilter("Kids", negative=True))
    recommender.add_recommendation_filter(filters.MediaTypeFilter("tv"))

    return recommender


if __name__ == "__main__":
    recommender = create_recommender()

    recommendations = recommender.recommend_seasonal_anime_for_user(
        "Janiskeisari", "2023", "winter"
    )
    print(recommendations[0:25])
