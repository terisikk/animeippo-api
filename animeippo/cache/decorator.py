import functools


def cached_query(ttl):
    def decorator_query(func):
        @functools.wraps(func)
        async def wrapper(self, query, parameters):
            data = None
            cachekey = "".join(query.split()) + str(parameters)

            if self.cache and self.cache.is_available():
                data = self.cache.get_json(cachekey)

            if data:
                print(f"Cache hit for {cachekey}")
                return data
            else:
                print(f"Cache miss for {cachekey}")
                data = await func(self, query, parameters)

                if self.cache and self.cache.is_available():
                    print(f"Cache save for {cachekey}")
                    self.cache.set_json(cachekey, data, ttl)

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

            if self.cache and self.cache.is_available():
                data = self.cache.get_dataframe(cachekey)

            if data is not None:
                print(f"Cache hit for {cachekey}")
                return data
            else:
                print(f"Cache miss for {cachekey}")
                data = await func(self, *args)

                if self.cache and self.cache.is_available():
                    print(f"Cache save for {cachekey}")
                    self.cache.set_dataframe(cachekey, data, ttl)

            return data

        return wrapper

    return decorator_query
