import asyncio
import functools
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


def cached_query(ttl):
    def decorator_query(func):
        @functools.wraps(func)
        async def wrapper(self, query, parameters):
            data = None
            cachekey = "".join(query.split()) + str(parameters)

            cache_available = self.cache is not None and self.cache.is_available()

            if cache_available:
                data = await asyncio.to_thread(self.cache.get_json, cachekey)

            if data:
                logger.debug(f"Cache hit for {cachekey}")
                return data
            else:
                logger.debug(f"Cache miss for {cachekey}")
                data = await func(self, query, parameters)

                if cache_available:
                    logger.debug(f"Cache save for {cachekey}")
                    threading.Thread(
                        target=self.cache.set_json,
                        args=(
                            cachekey,
                            data,
                            ttl,
                        ),
                    ).start()

            return data

        return wrapper

    return decorator_query


def cached_dataframe(ttl):
    def decorator_query(func):
        @functools.wraps(func)
        async def wrapper(self, *args):
            data = None
            cachekey = (
                func.__name__
                + " "
                + ",".join([str(arg) if arg else "" for arg in args])
                + "_"
                + str(self.__class__)
            )

            cache_available = self.cache is not None and self.cache.is_available()

            if cache_available:
                data = await asyncio.to_thread(self.cache.get_dataframe, cachekey)

            if data is not None:
                logger.debug(f"Cache hit for {cachekey}")
                return data
            else:
                logger.debug(f"Cache miss for {cachekey}")
                data = await func(self, *args)

                if cache_available:
                    logger.debug(f"Cache save for {cachekey}")

                    with ThreadPoolExecutor() as executor:
                        executor.submit(
                            self.cache.set_dataframe,
                            cachekey,
                            data,
                            ttl,
                        )

            return data

        return wrapper

    return decorator_query
