from flask import Flask, Response, request
from flask_cors import CORS

from animeippo.view import views
from animeippo.recommendation import builder, util as pdutil


app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
cors = CORS(app, origins="http://localhost:3000")

recommender = builder.create_builder("anilist").build()


@app.route("/api/seasonal")
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


@app.route("/api/recommend")
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


@app.route("/api/analyze")
def analyze_profile():
    user = request.args.get("user", None)

    if user is None:
        return "Validation error", 400

    dataset = recommender.async_get_dataset("2023", "spring", user)
    dataset = recommender.engine.fit(dataset)

    gdf = dataset.watchlist.explode("features")
    descriptions = pdutil.extract_features(gdf["features"], gdf["cluster"], 2)

    categories = [
        {"name": " ".join(descriptions.iloc[key].tolist()), "items": value.tolist()}
        for key, value in dataset.watchlist.sort_values("title").groupby("cluster").groups.items()
    ]

    return Response(views.web_view(dataset.watchlist, categories), mimetype="application/json")
