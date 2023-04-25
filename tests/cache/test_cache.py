import animeippo.cache.redis_cache as cache
import animeippo.providers.myanimelist as mal

from tests import test_data


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

    def json(self):
        return self.store

    def expire(self, key, ttl):
        pass


def test_items_can_be_added_to_redis_cache(mocker):
    mocker.patch("redis.Redis", RedisStub)

    r = cache.RedisCache()

    item = {"test": 1, "neste": [{"json": 2}]}

    key = "test"

    r.set_json(key, item)

    assert r.get_json(key) == item


def test_mal_can_fetch_values_from_cache(mocker, requests_mock):
    mocker.patch("redis.Redis", RedisStub)

    rcache = cache.RedisCache()

    provider = mal.MyAnimeListProvider(rcache)

    year = "2023"
    season = "winter"

    rcache.set_json("fake", test_data.MAL_SEASONAL_LIST)

    url = f"{mal.MAL_API_URL}/anime/season/{year}/{season}"
    adapter = requests_mock.get(url, json=None)  # nosec B113

    seasonal_list = provider.get_seasonal_anime_list(year, season)

    assert not adapter.called
    assert seasonal_list["title"].tolist() == [
        "Golden Kamuy 4th Season",
        "Shingeki no Kyojin: The Final Season",
    ]


def test_mal_related_anime_can_use_cache(mocker, requests_mock):
    mocker.patch("redis.Redis", RedisStub)

    rcache = cache.RedisCache()

    provider = mal.MyAnimeListProvider(rcache)

    id = "30"
    rcache.set_json("fake", {"data": test_data.MAL_RELATED_ANIME["related_anime"]})

    url = f"{mal.MAL_API_URL}/anime/{id}"
    adapter = requests_mock.get(url, json=None)  # nosec B113

    related_anime = provider.get_related_anime(id)

    assert related_anime.index.tolist() == [31]
    assert not adapter.called


def test_mal_list_can_be_stored_to_cache(mocker, requests_mock):
    mocker.patch("redis.Redis", RedisStub)

    rcache = cache.RedisCache()

    provider = mal.MyAnimeListProvider(rcache)

    year = "2023"
    season = "winter"

    url = f"{mal.MAL_API_URL}/anime/season/{year}/{season}"
    adapter = requests_mock.get(url, json=test_data.MAL_SEASONAL_LIST)  # nosec B113

    first_hit = provider.get_seasonal_anime_list(year, season)
    assert adapter.call_count == 1

    second_hit = provider.get_seasonal_anime_list(year, season)
    assert adapter.call_count == 1
    assert not first_hit.empty
    assert first_hit["title"].tolist() == second_hit["title"].tolist()


def test_mal_related_anime_can_be_stored_to_cache(mocker, requests_mock):
    mocker.patch("redis.Redis", RedisStub)

    rcache = cache.RedisCache()

    provider = mal.MyAnimeListProvider(rcache)

    id = "30"

    url = f"{mal.MAL_API_URL}/anime/{id}"
    adapter = requests_mock.get(url, json=test_data.MAL_RELATED_ANIME)  # nosec B113

    first_hit = provider.get_related_anime(id)
    assert adapter.call_count == 1

    second_hit = provider.get_related_anime(id)
    assert adapter.call_count == 1
    assert not first_hit.empty
    assert first_hit["title"].tolist() == second_hit["title"].tolist()
