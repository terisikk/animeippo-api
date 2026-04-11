import enum
import hashlib
from datetime import timedelta

import polars as pl
import redis


class CacheMode(enum.Enum):
    READ_WRITE = "read_write"
    WRITE_ONLY = "write_only"


class RedisCache:
    """We are wrapping the redis client to provide a stable interface
    in case we want to switch the caching solution."""

    def __init__(self, mode=CacheMode.READ_WRITE):
        # TODO: Remove this hardcoded server value, add to config
        self.connection = redis.Redis(host="redis-stack-server", port=6379)
        self.mode = mode

    def set_json(self, key, value, ttl=timedelta(days=7)):
        # We are using query strings as keys, better to hash them for perf
        key = hashlib.sha256(key.encode("utf-8")).hexdigest()

        self.connection.json().set(key, "$", value)
        self.connection.expire(key, ttl)

    def get_json(self, key):
        if self.mode == CacheMode.WRITE_ONLY:
            return None

        key = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.connection.json().get(key)

    def set_dataframe(self, key, dataframe, ttl=timedelta(days=7)):
        if dataframe is not None:
            self.connection.set(key, dataframe.write_ipc(None).getvalue())
            self.connection.expire(key, ttl)

    def get_dataframe(self, key):
        if self.mode == CacheMode.WRITE_ONLY:
            return None

        data = self.connection.get(key)

        return pl.read_ipc(data) if data is not None else None

    def is_available(self):
        try:
            self.connection.ping()
        except (redis.exceptions.ConnectionError, ConnectionRefusedError):
            return False
        return True
