import logging
import os

from aiohttp.client_exceptions import ClientError
from dotenv import load_dotenv
from flask import Flask, Response, request
from flask_cors import CORS

from animeippo.profiling import analyser
from animeippo.profiling.characteristics import Characteristics
from animeippo.recommendation import recommender_builder
from animeippo.view import views

# Load environment variables from .env file
load_dotenv("conf/prod.env")

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
cors = CORS(app, origins="http://localhost:3000")

# Read debug mode from environment
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Read log level from environment (defaults based on DEBUG mode)
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG" if DEBUG else "INFO").upper()

# Map string to logging level, with fallback to INFO for invalid values
log_level_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
log_level = log_level_map.get(LOG_LEVEL, logging.INFO)

logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

mode = "DEBUG" if DEBUG else "PRODUCTION"
logger.info(f"Starting AnimeIppo in {mode} mode (log level: {LOG_LEVEL})")

recommenders = {
    "anilist": recommender_builder.build_recommender("anilist"),
    "mixed": recommender_builder.build_recommender("mixed"),
}

profilers = {name: analyser.ProfileAnalyser(rec.provider) for name, rec in recommenders.items()}


def get_provider_instances(request):
    provider = request.args.get("provider", "anilist")
    return recommenders.get(provider, recommenders["anilist"]), profilers.get(
        provider, profilers["anilist"]
    )


@app.before_request
def log_request():
    """Log incoming requests with parameters."""
    params = dict(request.args)
    logger.info(f"{request.method} {request.path} - params: {params}")


@app.after_request
def log_response(response):
    """Log response status."""
    logger.info(f"{request.method} {request.path} - status: {response.status_code}")
    return response


@app.route("/seasonal")
def seasonal_anime():
    """Returns a json-list of seasonal or yearly anime titles."""
    year = request.args.get("year", None)
    season = request.args.get("season", None)

    if not year:
        return "Validation error", 400

    recommender, _ = get_provider_instances(request)
    dataset = recommender.recommend_seasonal_anime(year, season)

    return Response(
        views.recommendations_web_view(dataset.seasonal),
        mimetype="application/json",
    )


@app.route("/recommend")
def recommend_anime():
    """Recommends new anime to a user, either from a year or a single season.
    Accepts provider parameter: 'anilist' (default) or 'mixed' (for MAL users).
    """
    user = request.args.get("user", None)
    year = request.args.get("year", None)
    season = request.args.get("season", None)
    only_categories = request.args.get("only_categories", None)

    if not all([user, year]):
        return "Validation error", 400

    recommender, _ = get_provider_instances(request)

    try:
        dataset = recommender.recommend_seasonal_anime(year, season, user)
        categories = recommender.get_categories(dataset)
    except ClientError:
        return f"Could not fetch data for user {user}.", 404

    return Response(
        views.recommendations_web_view(
            None if only_categories else dataset.recommendations,
            categories,
            list(set(dataset.all_features) - set(dataset.nsfw_tags)),
            debug=DEBUG,
        ),
        mimetype="application/json",
    )


@app.route("/analyse")
def analyze_profile():
    """Analyses a user profile and clusters the watchlist.
    Accepts provider parameter: 'anilist' (default) or 'mixed' (for MAL users).
    """
    user = request.args.get("user", None)
    year = request.args.get("year", None)
    season = request.args.get("season", None)

    if user is None:
        return "Validation error", 400

    _, profiler = get_provider_instances(request)

    categories = profiler.analyse(user, year=year, season=season)

    return Response(
        views.profile_cluster_web_view(
            profiler.profile.watchlist.sort("title"),
            sorted(categories, key=lambda item: len(item["items"]), reverse=True),
            seasonal=profiler.seasonal,
        ),
        mimetype="application/json",
    )


@app.route("/profile")
def profile_characteristics():
    """Analyses a user profile and returns statistics.
    Accepts provider parameter: 'anilist' (default) or 'mixed' (for MAL users).
    """
    user = request.args.get("user", None)

    if user is None:
        return "Validation error", 400

    _, profiler = get_provider_instances(request)

    profiler.analyse(user)
    profiler.profile.characteristics = Characteristics(
        profiler.profile.watchlist, profiler.provider.get_genres()
    )

    genre_variance = profiler.profile.characteristics.genre_variance
    logger.debug(f"Profile genre variance for {user}: {genre_variance}")

    return Response(
        views.profile_characteristics_web_view(
            profiler.profile,
        ),
        mimetype="application/json",
    )
