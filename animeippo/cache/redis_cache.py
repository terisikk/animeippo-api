import io

import redis
import hashlib
import polars as pl
import polars.selectors as cs

from datetime import timedelta


class RedisCache:
    """We are wrapping the redis client to provide a stable interface
    in case we want to switch the caching solution."""

    def __init__(self):
        # TODO: Remove this hardcoded server value, add to config
        self.connection = redis.Redis(host="redis-stack-server", port=6379)

    def set_json(self, key, value, ttl=timedelta(days=7)):
        # We are using query strings as keys, better to hash them for perf
        key = hashlib.sha256(key.encode("utf-8")).hexdigest()

        self.connection.json().set(key, "$", value)
        self.connection.expire(key, ttl)

    def get_json(self, key):
        key = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.connection.json().get(key)

    def set_dataframe(self, key, dataframe, ttl=timedelta(days=7)):
        if dataframe is not None:
            self.connection.json().set(
                # Trying to write a null column to json will result in an error
                # at least with polars 0.20
                key,
                "$",
                dataframe.select(~cs.by_dtype(pl.Null)).write_json(),
            )
            self.connection.expire(key, ttl)

    def get_dataframe(self, key):
        data = self.connection.json().get(key)
        return pl.read_json(io.StringIO(data)) if data is not None else None

    def is_available(self):
        try:
            self.connection.ping()
        except (redis.exceptions.ConnectionError, ConnectionRefusedError):
            return False
        return True
