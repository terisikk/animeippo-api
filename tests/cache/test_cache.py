import animeippo.providers.myanimelist as mal
import pandas as pd

from animeippo import cache
from tests import test_data

import redis

import pytest


class ResponseStub:
    dictionary = {}

    def __init__(self, dictionary):
        self.dictionary = dictionary

    async def get(self, key):
        return self.dictionary.get(key)

    async def json(self):
        return self.dictionary

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def __aenter__(self):
        return self

    def raise_for_status(self):
        pass


class RedisJsonStub:
    def __init__(self, *args, **kwargs):
        self.store = None

    def set(self, key, point, value):
        self.store = value

    def get(self, key):
        return self.store


class RedisStub:
    def __init__(self, *args, **kwargs):
        self.store = RedisJsonStub()
        self.plainstore = {}
        self.available = True

    def json(self):
        return self.store

    def expire(self, key, ttl):
        pass

    def get(self, key):
        return self.plainstore.get(key, None)

    def set(self, key, data):
        self.plainstore[key] = data

    def ping(self):
        if self.available:
            return True
        else:
            raise redis.exceptions.ConnectionError()


def test_items_can_be_added_to_redis_cache(mocker):
    mocker.patch("redis.Redis", RedisStub)

    r = cache.RedisCache()

    item = {"test": 1, "neste": [{"json": 2}]}

    key = "test"

    r.set_json(key, item)

    assert r.get_json(key) == item


@pytest.mark.asyncio
async def test_mal_can_fetch_values_from_cache(mocker):
    mocker.patch("redis.Redis", RedisStub)

    rcache = cache.RedisCache()

    provider = mal.MyAnimeListProvider(rcache)

    year = "2023"
    season = "winter"

    rcache.set_json("fake", test_data.MAL_SEASONAL_LIST)

    response = ResponseStub({"data": {}})
    mocker.patch("aiohttp.ClientSession.get", return_value=response)

    seasonal_list = await provider.get_seasonal_anime_list(year, season)

    assert seasonal_list["title"].tolist() == [
        "Golden Kamuy 4th Season",
        "Shingeki no Kyojin: The Fake Season",
    ]


@pytest.mark.asyncio
async def test_mal_related_anime_can_use_cache(mocker):
    mocker.patch("redis.Redis", RedisStub)

    rcache = cache.RedisCache()

    provider = mal.MyAnimeListProvider(rcache)

    id = "30"
    rcache.set_json("fake", {"data": test_data.MAL_RELATED_ANIME["related_anime"]})

    response = ResponseStub(None)
    mocker.patch("aiohttp.ClientSession.get", return_value=response)

    related_anime = await provider.get_related_anime(id)

    assert related_anime == [31]


@pytest.mark.asyncio
async def test_mal_list_can_be_stored_to_cache(mocker):
    mocker.patch("redis.Redis", RedisStub)

    rcache = cache.RedisCache()

    provider = mal.MyAnimeListProvider(rcache)

    year = "2023"
    season = "winter"

    response = ResponseStub(test_data.MAL_SEASONAL_LIST)
    mocker.patch("aiohttp.ClientSession.get", side_effect=[response, None])

    first_hit = await provider.get_seasonal_anime_list(year, season)

    second_hit = await provider.get_seasonal_anime_list(year, season)
    assert not first_hit.empty
    assert first_hit["title"].tolist() == second_hit["title"].tolist()


@pytest.mark.asyncio
async def test_mal_related_anime_can_be_stored_to_cache(mocker):
    mocker.patch("redis.Redis", RedisStub)

    rcache = cache.RedisCache()

    provider = mal.MyAnimeListProvider(rcache)

    id = "30"

    response = ResponseStub(test_data.MAL_RELATED_ANIME)
    mocker.patch("aiohttp.ClientSession.get", side_effect=[response, None])

    first_hit = await provider.get_related_anime(id)

    second_hit = await provider.get_related_anime(id)
    assert len(first_hit) != 0
    assert first_hit == second_hit


def test_dataframes_can_be_added_to_cache(mocker):
    mocker.patch("redis.Redis", RedisStub)

    rcache = cache.RedisCache()

    data = pl.DataFrame(test_data.FORMATTED_MAL_USER_LIST)

    rcache.set_dataframe("test", data)

    actual = rcache.get_dataframe("test")

    assert actual["title"].tolist() == data["title"].tolist()
    assert actual.columns.tolist() == data.columns.tolist()


def test_dicts_are_parsed_correctly_when_reading_from_cache(mocker):
    mocker.patch("redis.Redis", RedisStub)

    rcache = cache.RedisCache()

    data = pl.DataFrame(
        {
            "ranks": [
                {
                    "rank1": 1,
                    "rank2": 2,
                },
                {
                    "rank3": 3,
                    "rank4": 4,
                },
            ]
        }
    )

    rcache.set_dataframe("test", data)

    actual = rcache.get_dataframe("test")

    assert actual["ranks"].tolist() == data["ranks"].tolist()


def test_none_frames_are_not_added_to_cache(mocker):
    mocker.patch("redis.Redis", RedisStub)

    rcache = cache.RedisCache()

    data = None

    rcache.set_dataframe("test", data)

    assert rcache.connection.plainstore == {}


@pytest.mark.asyncio
async def test_data_can_be_fetched_even_with_cache_connection_error(mocker):
    mocker.patch("redis.Redis", RedisStub)

    rcache = cache.RedisCache()
    rcache.connection.available = False

    provider = mal.MyAnimeListProvider(rcache)

    year = "2023"
    season = "winter"

    response = ResponseStub(test_data.MAL_SEASONAL_LIST)
    mocker.patch("aiohttp.ClientSession.get", side_effect=[response, None])

    seasonal_list = await provider.get_seasonal_anime_list(year, season)

    assert seasonal_list["title"].tolist() == [
        "Golden Kamuy 4th Season",
        "Shingeki no Kyojin: The Fake Season",
    ]
