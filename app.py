import time

from flask import Flask, Response, request, g
from flask_cors import CORS

from animeippo.main import create_recommender
from animeippo.recommendation import filters


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


recommender_engine = create_recommender()


@app.route("/api/seasonal")
def seasonal_anime():
    year = request.args.get("year", None)
    season = request.args.get("season", None)

    if not all([year, season]):
        return "Validation error", 400

    seasonal = recommender_engine.provider.get_seasonal_anime_list(year, season)

    seasonal_filters = [
        filters.GenreFilter("Kids", negative=True),
        filters.MediaTypeFilter("tv", "movie"),
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

    recommendations = recommender_engine.recommend_seasonal_anime_for_user(user, year, season)

    recommendations["id"] = recommendations.index

    return Response(recommendations.to_json(orient="records"), mimetype="application/json")
