class ResponseStub:
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


class SessionStub:
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def get(self, key, *args, **kwargs):
        return self.dictionary.get(key)

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def __aenter__(self):
        return self
