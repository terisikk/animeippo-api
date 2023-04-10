from animeippo.providers import myanimelist as mal
from animeippo.recommendation import engine, filters, scoring


def create_recommender():
    provider = mal.MyAnimeListProvider()
    encoder = scoring.CategoricalEncoder(provider.get_genre_tags())

    recommender = engine.AnimeRecommendationEngine(provider)

    scorers = [
        scoring.GenreAverageScorer(encoder),
        scoring.GenreSimilarityScorer(encoder, weighted=True),
        scoring.ClusterSimilarityScorer(encoder, weighted=True),
        scoring.StudioCountScorer(),
        scoring.StudioAverageScorer(weighted=True),
    ]

    for scorer in scorers:
        recommender.add_scorer(scorer)

    recfilters = [filters.GenreFilter("Kids", negative=True), filters.MediaTypeFilter("tv")]

    for filter in recfilters:
        recommender.add_recommendation_filter(filter)

    return recommender


if __name__ == "__main__":
    recommender = create_recommender()

    recommendations = recommender.recommend_seasonal_anime_for_user("Nemoria", "2023", "winter")
    print(recommendations[0:25].drop(["media_type", "id", "user_score"], axis=1))
