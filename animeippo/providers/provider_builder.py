import asyncio

from async_lru import alru_cache
from animeippo.providers import filters
from animeippo.recommendation import model, profile

import polars as pl


@alru_cache(maxsize=1)
async def get_user_profile(provider, user):
    if user is None:
        return None

    user_data = await provider.get_user_anime_list(user)
    return profile.UserProfile(user, user_data)


@alru_cache(maxsize=1)
async def get_user_manga_list(provider, user):
    if user is None:
        return None

    return await provider.get_user_manga_list(user)


async def get_related_anime(indices, provider):
    return [await provider.get_related_anime(index) for index in indices]


async def get_dataset(provider, user, year, season):
    user_profile, manga_data, season_data = await asyncio.gather(
        get_user_profile(provider, user),
        get_user_manga_list(provider, user),
        provider.get_seasonal_anime_list(year, season),
    )

    if user_profile is not None:
        user_profile.mangalist = manga_data

    data = model.RecommendationModel(user_profile, season_data)
    data.nsfw_tags = provider.get_nsfw_tags()

    return data


def fill_user_status_data_from_watchlist(seasonal, watchlist):
    return seasonal.join(watchlist.select(["id", "user_status"]), on="id", how="left")


async def fit_data(data, seasonal_filters, provider, fetch_related_anime=False):
    if data.seasonal is not None:
        if seasonal_filters:
            data.seasonal = data.seasonal.filter(seasonal_filters)

        if fetch_related_anime:
            indices = data.seasonal["id"].to_list()
            related_anime = await get_related_anime(indices, provider)
            data.seasonal.with_columns(continuation_to=related_anime)

    if data.seasonal is not None and data.watchlist is not None:
        data.seasonal = fill_user_status_data_from_watchlist(data.seasonal, data.watchlist)
        data.seasonal = (
            data.seasonal.filter(filters.ContinuationFilter(data.watchlist))
            if data.seasonal["continuation_to"].dtype != pl.List(pl.Null)
            else data.seasonal
        )

    return data


async def construct_anilist_data(provider, year, season, user):
    data = await get_dataset(provider, user, year, season)

    return await fit_data(data, None, provider)


async def construct_myanimelist_data(provider, year, season, user):
    data = await get_dataset(provider, user, year, season)

    seasonal_filters = [
        filters.RatingFilter("g", "rx", negative=True),
        filters.StartSeasonFilter([int(year)], [season]),
    ]

    return await fit_data(data, seasonal_filters, provider, fetch_related_anime=True)
