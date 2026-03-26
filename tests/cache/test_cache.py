import polars as pl
import pytest
import redis

from animeippo import cache
from animeippo.providers.myanimelist.connection import MyAnimeListConnection
from tests import test_data


class ResponseStub:
    def __init__(self, dictionary):
        self.dictionary = dictionary
        self.status = 200

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
async def test_connection_can_fetch_values_from_cache(mocker):
    mocker.patch("redis.Redis", RedisStub)

    rcache = cache.RedisCache()
    connection = MyAnimeListConnection(rcache)

    rcache.set_json("fake", test_data.MAL_SEASONAL_LIST)

    response = ResponseStub({"data": {}})
    mocker.patch("aiohttp.ClientSession.get", return_value=response)

    seasonal_list = await connection.request_anime_list("fake", {})

    assert seasonal_list["data"][0]["node"]["title"] == "Golden Kamuy 4th Season"


@pytest.mark.asyncio
async def test_connection_can_store_to_cache(mocker):
    mocker.patch("redis.Redis", RedisStub)

    rcache = cache.RedisCache()
    connection = MyAnimeListConnection(rcache)

    response = ResponseStub(test_data.MAL_SEASONAL_LIST)
    mocker.patch("aiohttp.ClientSession.request", side_effect=[response, None])

    first_hit = await connection.request_anime_list("fake_query", {})
    second_hit = await connection.request_anime_list("fake_query", {})

    assert first_hit == second_hit


def test_dataframes_can_be_added_to_cache(mocker):
    mocker.patch("redis.Redis", RedisStub)

    rcache = cache.RedisCache()

    data = pl.DataFrame(test_data.FORMATTED_MAL_USER_LIST)

    rcache.set_dataframe("test", data)

    actual = rcache.get_dataframe("test")

    assert actual["title"].to_list() == data["title"].to_list()
    assert actual.columns == data.columns


def test_none_frames_are_not_added_to_cache(mocker):
    mocker.patch("redis.Redis", RedisStub)

    rcache = cache.RedisCache()

    data = None

    rcache.set_dataframe("test", data)

    assert rcache.connection.plainstore == {}


@pytest.mark.asyncio
async def test_cached_dataframe_decorator_stores_and_retrieves(mocker):
    from datetime import timedelta

    from animeippo.providers import caching

    mocker.patch("redis.Redis", RedisStub)

    rcache = cache.RedisCache()

    class FakeProvider:
        def __init__(self):
            self.cache = rcache
            self.call_count = 0

        @caching.cached_dataframe(ttl=timedelta(days=1))
        async def get_data(self, key):
            self.call_count += 1
            return pl.DataFrame({"id": [1, 2], "title": ["A", "B"]})

    provider = FakeProvider()

    first = await provider.get_data("test")
    second = await provider.get_data("test")

    assert first["title"].to_list() == ["A", "B"]
    assert second["title"].to_list() == ["A", "B"]


@pytest.mark.asyncio
async def test_data_can_be_fetched_even_with_cache_connection_error(mocker):
    mocker.patch("redis.Redis", RedisStub)

    rcache = cache.RedisCache()
    rcache.connection.available = False

    connection = MyAnimeListConnection(rcache)

    response = ResponseStub(test_data.MAL_SEASONAL_LIST)
    mocker.patch("aiohttp.ClientSession.request", side_effect=[response, None])

    result = await connection.request_anime_list("fake_query", {})

    assert result["data"][0]["node"]["title"] == "Golden Kamuy 4th Season"
