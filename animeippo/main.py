import animeippo.providers as providers
from animeippo.recommendation import engine, filters, scoring, dataset
from animeippo import cache


def create_recommender():
    scorers = [
        # redundant
        # scoring.GenreSimilarityScorer(feature_tags, weighted=True),
        scoring.GenreAverageScorer(),
        scoring.ClusterSimilarityScorer(weighted=True),
        # redundant
        # scoring.StudioCountScorer(),
        scoring.StudioAverageScorer(),
        scoring.PopularityScorer(),
        scoring.ContinuationScorer(),
        scoring.SourceScorer(),
        scoring.DirectSimilarityScorer(),
    ]

    recommender = engine.AnimeRecommendationEngine(scorers)

    return recommender


def create_user_dataset(user, year, season, provider):
    data = dataset.UserDataSet(
        provider.get_user_anime_list(user),
        provider.get_seasonal_anime_list(year, season),
        provider.get_features(),
    )

    watchlist_filters = [filters.IdFilter(*data.seasonal.index.to_list(), negative=True)]

    for f in watchlist_filters:
        data.watchlist = f.filter(data.watchlist)

    seasonal_filters = [
        filters.FeatureFilter("Kids", negative=True),
        filters.FeatureFilter("Hentai", negative=True),
        # filters.MediaTypeFilter("tv"),
        # filters.RatingFilter("g", "rx", negative=True),
        filters.StartSeasonFilter((year, season)),
    ]

    for f in seasonal_filters:
        data.seasonal = f.filter(data.seasonal)

    if "related_anime" not in data.seasonal.columns:
        related_anime = data.seasonal.index.map(
            lambda i: provider.get_related_anime(i).index.to_list()
        )
        data.seasonal["related_anime"] = related_anime.to_list()

    data.seasonal = filters.ContinuationFilter(data.watchlist).filter(data.seasonal)

    return data


if __name__ == "__main__":
    year = "2022"
    season = "spring"
    user = "Janiskeisari"

    provider = providers.anilist.AniListProvider(cache=cache.RedisCache())

    data = create_user_dataset(user, year, season, provider)
    recommender = create_recommender()

    recommendations = recommender.fit_predict(data)
    print(recommendations.reset_index().loc[0:25, ["title", "features", "recommend_score"]])
