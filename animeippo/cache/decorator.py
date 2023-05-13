import functools


def cached_query(ttl):
    def decorator_query(func):
        @functools.wraps(func)
        def wrapper(self, query, parameters):
            data = None
            cachekey = "".join(query.split()) + str(parameters)

            if self.cache:
                data = self.cache.get_json(cachekey)

            if data:
                print(f"Cache hit for {cachekey}")
                return data
            else:
                print(f"Cache miss for {cachekey}")
                data = func(self, query, parameters)

                if self.cache:
                    print(f"Cache save for {cachekey}")
                    self.cache.set_json(cachekey, data, ttl)

            return data

        return wrapper

    return decorator_query
