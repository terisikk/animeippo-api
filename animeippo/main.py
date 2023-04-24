from animeippo.providers import myanimelist as mal
from animeippo.recommendation import engine, filters, scoring


def create_recommender():
    provider = mal.MyAnimeListProvider()
    encoder = scoring.CategoricalEncoder(provider.get_genre_tags())

    recommender = engine.AnimeRecommendationEngine(provider)

    scorers = [
        # redundant
        # scoring.GenreSimilarityScorer(encoder, weighted=True),
        scoring.GenreAverageScorer(),
        scoring.ClusterSimilarityScorer(encoder, weighted=True),
        # redundant
        # scoring.StudioCountScorer(),
        scoring.StudioAverageScorer(),
        scoring.PopularityScorer(),
    ]

    for scorer in scorers:
        recommender.add_scorer(scorer)

    recfilters = [
        filters.GenreFilter("Kids", negative=True),
        filters.MediaTypeFilter("tv", "movie"),
        filters.RatingFilter("g", "rx", negative=True),
    ]

    for filter in recfilters:
        recommender.add_recommendation_filter(filter)

    return recommender


if __name__ == "__main__":
    year = "2023"
    season = "spring"

    recommendations = engine.recommend_seasonal_anime_for_user("Janiskeisari", year, season)
    print(recommendations.reset_index().loc[0:25, ["title", "genres", "mean", "recommend_score"]])
