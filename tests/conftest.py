import pytest


class MockRedisCache:
    """Mock Redis cache that always reports as unavailable during tests.
    This ensures consistent test coverage regardless of whether Redis is running."""

    def is_available(self):
        return False

    def set_json(self, key, value, ttl=None):
        pass

    def get_json(self, key):
        return None

    def set_dataframe(self, key, dataframe, ttl=None):
        pass

    def get_dataframe(self, key):
        return None


@pytest.fixture(autouse=True)
def mock_redis_cache(request, monkeypatch):
    """Automatically mock RedisCache for all tests to ensure consistent coverage.

    Skip this fixture for tests in the cache module since they have their own Redis mocks.
    """
    # Don't apply to cache tests - they have their own Redis mocks
    if "cache/test_cache" in request.node.nodeid:
        return

    monkeypatch.setattr("animeippo.cache.RedisCache", MockRedisCache)
