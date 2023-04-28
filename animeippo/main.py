from animeippo.providers import myanimelist as mal
from animeippo.recommendation import engine, filters, scoring, dataset
from animeippo import cache


def create_recommender(genre_tags):
    scorers = [
        # redundant
        # scoring.GenreSimilarityScorer(genre_tags, weighted=True),
        scoring.GenreAverageScorer(),
        scoring.ClusterSimilarityScorer(genre_tags, weighted=True),
        # redundant
        # scoring.StudioCountScorer(),
        scoring.StudioAverageScorer(),
        scoring.PopularityScorer(),
        scoring.ContinuationScorer(),
        scoring.SourceScorer(),
    ]

    recommender = engine.AnimeRecommendationEngine(scorers)

    return recommender


def create_user_dataset(user, year, season, provider):
    data = dataset.UserDataSet(
        provider.get_user_anime_list(user), provider.get_seasonal_anime_list(year, season)
    )

    watchlist_filters = [filters.IdFilter(*data.seasonal.index.to_list(), negative=True)]

    for f in watchlist_filters:
        data.watchlist = f.filter(data.watchlist)

    seasonal_filters = [
        filters.GenreFilter("Kids", negative=True),
        filters.MediaTypeFilter("tv"),
        filters.RatingFilter("g", "rx", negative=True),
        filters.StartSeasonFilter((year, season)),
    ]

    for f in seasonal_filters:
        data.seasonal = f.filter(data.seasonal)

    related_anime = data.seasonal.index.map(lambda i: provider.get_related_anime(i).index.to_list())
    data.seasonal["related_anime"] = related_anime.to_list()

    data.seasonal = filters.ContinuationFilter(data.watchlist).filter(data.seasonal)

    return data


if __name__ == "__main__":
    year = "2023"
    season = "winter"
    user = "Janiskeisari"

    provider = mal.MyAnimeListProvider(cache=cache.RedisCache())

    data = create_user_dataset(user, year, season, provider)
    recommender = create_recommender(provider.get_genre_tags())

    recommendations = recommender.fit_predict(data)
    print(recommendations.reset_index().loc[0:25, ["title", "genres", "mean", "recommend_score"]])
