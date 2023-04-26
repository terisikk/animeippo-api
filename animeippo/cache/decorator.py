import functools


def cached_query(ttl):
    def decorator_query(func):
        @functools.wraps(func)
        def wrapper(self, query, parameters):
            data = None

            if self.cache:
                data = self.cache.get_json(query + str(parameters))

            if data:
                return data
            else:
                data = func(self, query, parameters)

                if self.cache:
                    self.cache.set_json(query + str(parameters), data, ttl)

            return data

        return wrapper

    return decorator_query
