import time

from animeippo.providers import myanimelist as mal
from animeippo.recommendation import engine, filters, scoring

from flask import Flask, Response, request, g
from flask_cors import CORS


def create_recommender():
    provider = mal.MyAnimeListProvider()
    encoder = scoring.CategoricalEncoder(provider.get_genre_tags())

    recommender = engine.AnimeRecommendationEngine(provider)

    scorers = [
        # scoring.GenreSimilarityScorer(encoder, weighted=True),
        scoring.GenreAverageScorer(),
        scoring.ClusterSimilarityScorer(encoder, weighted=True),
        # scoring.StudioCountScorer(),
        scoring.StudioAverageScorer(),
        scoring.PopularityScorer(),
    ]

    for scorer in scorers:
        recommender.add_scorer(scorer)

    recfilters = [
        filters.GenreFilter("Kids", negative=True),
        filters.MediaTypeFilter("tv", "movie"),
        # filters.StatusFilter("dropped", "on_hold", negative=True),
        filters.RatingFilter("g", "rx", negative=True),
    ]

    for filter in recfilters:
        recommender.add_recommendation_filter(filter)

    return recommender


app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
cors = CORS(app, origins="http://localhost:3000")


@app.before_request
def before_request():
    g.start = time.time()


@app.after_request
def after_request(response):
    diff = time.time() - g.start
    print(diff)
    return response


engine = create_recommender()


@app.route("/api/seasonal")
def seasonal_anime():
    year = request.args.get("year", None)
    season = request.args.get("season", None)

    if not all([year, season]):
        return "Validation error", 400

    seasonal = engine.provider.get_seasonal_anime_list(year, season)

    seasonal_filters = [
        filters.GenreFilter("Kids", negative=True),
        filters.MediaTypeFilter("tv", "movie"),
        # filters.StatusFilter("dropped", "on_hold", negative=True),
        filters.RatingFilter("g", "rx", negative=True),
        filters.StartSeasonFilter((year, season)),
    ]

    for f in seasonal_filters:
        seasonal = f.filter(seasonal)

    return Response(
        seasonal.sort_values("num_list_users", ascending=False).to_json(orient="records"),
        mimetype="application/json",
    )


@app.route("/api/recommend")
def recommend_anime():
    user = request.args.get("user", None)
    year = request.args.get("year", None)
    season = request.args.get("season", None)

    if not all([user, year, season]):
        return "Validation error", 400

    recommendations = engine.recommend_seasonal_anime_for_user(user, year, season)

    recommendations["id"] = recommendations.index

    return Response(recommendations.to_json(orient="records"), mimetype="application/json")


if __name__ == "__main__":
    year = "2023"
    season = "spring"

    recommendations = engine.recommend_seasonal_anime_for_user("Janiskeisari", year, season)
    print(recommendations.reset_index().loc[0:25, ["title", "genres", "mean", "recommend_score"]])
