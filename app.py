import time

from flask import Flask, Response, request, g
from flask_cors import CORS

from animeippo.main import create_recommender, create_user_dataset
from animeippo.providers import myanimelist as mal
from animeippo.cache import redis_cache
from animeippo.recommendation import filters


app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
cors = CORS(app, origins="http://localhost:3000")

provider = mal.MyAnimeListProvider(cache=redis_cache.RedisCache())
recommender_engine = create_recommender(provider.get_genre_tags())


@app.before_request
def before_request():
    g.start = time.time()


@app.after_request
def after_request(response):
    diff = time.time() - g.start
    print(diff)
    return response


@app.route("/api/seasonal")
def seasonal_anime():
    year = request.args.get("year", None)
    season = request.args.get("season", None)

    if not all([year, season]):
        return "Validation error", 400

    seasonal = provider.get_seasonal_anime_list(year, season)

    seasonal_filters = [
        filters.MediaTypeFilter("tv", "movie"),
        filters.RatingFilter("rx", negative=True),
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

    dataset = create_user_dataset(user, year, season, provider)

    recommendations = recommender_engine.fit_predict(dataset)

    recommendations["id"] = recommendations.index

    return Response(recommendations.to_json(orient="records"), mimetype="application/json")
