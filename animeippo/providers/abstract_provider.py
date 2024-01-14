import abc


class AbstractAnimeProvider(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_user_anime_list(self, user_id):
        pass

    @abc.abstractmethod
    def get_user_manga_list(self, user_id):
        pass

    @abc.abstractmethod
    def get_seasonal_anime_list(self, year, season):
        pass

    @abc.abstractmethod
    def get_feature_fields(self):
        pass

    @abc.abstractmethod
    def get_related_anime(self, id):
        pass
