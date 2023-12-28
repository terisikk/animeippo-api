import polars as pl

from tests import test_data


class AsyncProviderStub:
    def __init__(
        self,
        seasonal=test_data.FORMATTED_MAL_SEASONAL_LIST,
        user=test_data.FORMATTED_MAL_USER_LIST,
        manga=[],
        cache=None,
    ):
        self.seasonal = seasonal
        self.user = user
        self.manga = manga
        self.cache = cache

    async def get_seasonal_anime_list(self, *args, **kwargs):
        return pl.DataFrame(self.seasonal)

    async def get_user_anime_list(self, *args, **kwargs):
        return pl.DataFrame(self.user)

    async def get_user_manga_list(self, *args, **kwargs):
        return pl.DataFrame(self.manga)

    async def get_related_anime(self, *args, **kwargs):
        return pl.DataFrame()

    def get_features(self, *args, **kwargs):
        return ["genres"]


class FaultyProviderStub:
    def __init__(
        self,
        cache=None,
    ):
        self.cache = cache

    async def get_seasonal_anime_list(self, *args, **kwargs):
        return None

    async def get_user_anime_list(self, *args, **kwargs):
        return None

    async def get_user_manga_list(self, *args, **kwargs):
        return None

    async def get_related_anime(self, *args, **kwargs):
        return None

    def get_features(self, *args, **kwargs):
        return ["genres"]
