from flask import Flask, Response, request
from flask_cors import CORS

from animeippo.view import views
from animeippo.recommendation import recommender_builder, profile

from aiohttp.client_exceptions import ClientError

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
cors = CORS(app, origins="http://localhost:3000")

recommender = recommender_builder.create_builder("mixed").build()
profiler = profile.ProfileAnalyser(recommender.provider)


@app.route("/seasonal")
def seasonal_anime():
    """Returns a json-list of seasonal or yearly anime titles."""
    year = request.args.get("year", None)
    season = request.args.get("season", None)

    if not year:
        return "Validation error", 400

    dataset = recommender.recommend_seasonal_anime(year, season)

    return Response(
        views.web_view(dataset.seasonal.sort_values("popularity", ascending=False)),
        mimetype="application/json",
    )


@app.route("/recommend")
def recommend_anime():
    """Recommends new anime to a user, either from a year or a single season.
    Currently users are accepted only from Anilist. Returns a json-representation.
    """
    user = request.args.get("user", None)
    year = request.args.get("year", None)
    season = request.args.get("season", None)

    if not all([user, year]):
        return "Validation error", 400

    try:
        dataset = recommender.recommend_seasonal_anime(year, season, user)
        categories = recommender.get_categories(dataset)
    except ClientError:
        return f"Could nof fetch data for user {user}.", 404

    return Response(views.web_view(dataset.recommendations, categories), mimetype="application/json")


@app.route("/analyse")
def analyze_profile():
    """Analyses an Anilist user profile and clusters the watchlist to groups of
    simila anime with descriptions. Returns a json-representation."""
    user = request.args.get("user", None)

    if user is None:
        return "Validation error", 400

    categories = profiler.analyse(user)

    return Response(
        views.web_view(
            profiler.dataset.watchlist.sort_values(["title"]),
            sorted(categories, key=lambda item: len(item["items"]), reverse=True),
        ),
        mimetype="application/json",
    )
