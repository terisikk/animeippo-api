import os

import structlog
from aiohttp.client_exceptions import ClientError
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from animeippo.logging import configure_logging
from animeippo.profiling import analyser
from animeippo.profiling.characteristics import Characteristics
from animeippo.recommendation import recommender_builder
from animeippo.view import views

load_dotenv("conf/prod.env")

log_level = configure_logging()
logger = structlog.get_logger()

DEBUG = os.getenv("DEBUG", "false").lower() == "true"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("app_starting", mode="DEBUG" if DEBUG else "PRODUCTION", log_level=log_level)

recommenders = {
    "anilist": recommender_builder.build_recommender("anilist"),
    "mixed": recommender_builder.build_recommender("mixed"),
}

profilers = {
    name: analyser.ProfileAnalyser(
        rec.provider,
        clustering_defaults=recommender_builder.CLUSTERING_DEFAULTS,
    )
    for name, rec in recommenders.items()
}


def get_provider_instances(provider):
    return recommenders.get(provider, recommenders["anilist"]), profilers.get(
        provider, profilers["anilist"]
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        path=request.url.path,
        method=request.method,
        provider=request.query_params.get("provider", "anilist"),
        user=request.query_params.get("user"),
    )
    logger.info("request_started")
    response = await call_next(request)
    logger.info("request_completed", status=response.status_code)
    return response


@app.get("/seasonal")
async def seasonal_anime(
    year: str = Query(None),
    season: str = Query(None),
    provider: str = Query("anilist"),
):
    """Returns a json-list of seasonal or yearly anime titles."""
    if not year:
        return JSONResponse("Validation error", status_code=400)

    recommender, _ = get_provider_instances(provider)
    dataset = await recommender.recommend_seasonal_anime(year, season)

    return Response(
        content=views.recommendations_web_view(dataset.seasonal),
        media_type="application/json",
    )


@app.get("/recommend")
async def recommend_anime(
    user: str = Query(None),
    year: str = Query(None),
    season: str = Query(None),
    provider: str = Query("anilist"),
    only_categories: str = Query(None),
):
    """Recommends new anime to a user, either from a year or a single season.
    Accepts provider parameter: 'anilist' (default) or 'mixed' (for MAL users).
    """
    if not all([user, year]):
        return JSONResponse("Validation error", status_code=400)

    recommender, _ = get_provider_instances(provider)

    try:
        dataset = await recommender.recommend_seasonal_anime(year, season, user)
        categories = recommender.get_categories(dataset)
    except ClientError:
        logger.error("data_fetch_failed", exc_info=True)
        return JSONResponse(f"Could not fetch data for user {user}.", status_code=404)

    return Response(
        content=views.recommendations_web_view(
            None if only_categories else dataset.recommendations,
            categories,
            list(set(dataset.all_features) - set(dataset.nsfw_tags)),
            debug=DEBUG,
        ),
        media_type="application/json",
    )


@app.get("/analyse")
async def analyze_profile(
    user: str = Query(None),
    year: str = Query(None),
    season: str = Query(None),
    provider: str = Query("anilist"),
):
    """Analyses a user profile and clusters the watchlist.
    Accepts provider parameter: 'anilist' (default) or 'mixed' (for MAL users).
    """
    if user is None:
        return JSONResponse("Validation error", status_code=400)

    _, profiler = get_provider_instances(provider)

    profile, categories, seasonal = await profiler.analyse(user, year=year, season=season)

    return Response(
        content=views.profile_cluster_web_view(
            profile.watchlist.sort("title"),
            sorted(categories, key=lambda item: len(item["items"]), reverse=True),
            seasonal=seasonal,
        ),
        media_type="application/json",
    )


@app.get("/profile")
async def profile_characteristics(
    user: str = Query(None),
    provider: str = Query("anilist"),
):
    """Analyses a user profile and returns statistics.
    Accepts provider parameter: 'anilist' (default) or 'mixed' (for MAL users).
    """
    if user is None:
        return JSONResponse("Validation error", status_code=400)

    _, profiler = get_provider_instances(provider)

    profile, _, _ = await profiler.analyse(user)
    profile.characteristics = Characteristics(profile.watchlist, profiler.provider.get_genres())

    logger.debug(
        "profile_computed",
        genre_variance=profile.characteristics.genre_variance,
    )

    return Response(
        content=views.profile_characteristics_web_view(profile),
        media_type="application/json",
    )
