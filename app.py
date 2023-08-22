from flask import Flask, Response, request
from flask_cors import CORS

from animeippo.view import views
from animeippo.recommendation import recommender_builder, profile

import pandas as pd

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
cors = CORS(app, origins="http://localhost:3000")

recommender = recommender_builder.create_builder("anilist").build()
profiler = profile.ProfileAnalyser(recommender.provider)


@app.route("/seasonal")
def seasonal_anime():
    year = request.args.get("year", None)
    season = request.args.get("season", None)

    if not year:
        return "Validation error", 400

    seasonal = recommender.recommend_seasonal_anime(year, season)

    return Response(
        views.web_view(seasonal.sort_values("popularity", ascending=False)),
        mimetype="application/json",
    )


@app.route("/recommend")
def recommend_anime():
    user = request.args.get("user", None)
    year = request.args.get("year", None)
    season = request.args.get("season", None)

    if not all([user, year]):
        return "Validation error", 400

    dataset = recommender.recommend_seasonal_anime(year, season, user)
    categories = recommender.get_categories(dataset)

    return Response(
        views.web_view(dataset.recommendations, categories), mimetype="application/json"
    )


@app.route("/api/analyse")
def analyze_profile():
    user = request.args.get("user", None)

    if user is None:
        return "Validation error", 400

    categories = profiler.analyse(user)

    # TODO: Extract elsewhere. Possibly create categorical columns already in the formatters.
    profiler.dataset.watchlist["categorical_user_status"] = pd.Categorical(
        profiler.dataset.watchlist["user_status"],
        categories=["current", "planned", "completed", "paused", "dropped"],
    )

    return Response(
        views.web_view(
            profiler.dataset.watchlist.sort_values(["categorical_user_status", "title"]),
            sorted(categories, key=lambda item: len(item["items"]), reverse=True),
        ),
        mimetype="application/json",
    )
