import pandas as pd
from enum import StrEnum


class Columns(StrEnum):
    ID = "id"
    ID_MAL = "id_mal"
    TITLE = "title"
    FORMAT = "format"
    GENRES = "genres"
    TAGS = "tags"
    COVER_IMAGE = "cover_image"
    SCORE = "score"
    MEAN_SCORE = "mean_score"
    POPULARITY = "popularity"
    DURATION = "duration"
    EPISODES = "episodes"
    STATUS = "status"
    USER_STATUS = "user_status"
    SOURCE = "source"
    CONTINUATION_TO = "continuation_to"
    ADAPTATION_OF = "adaptation_of"
    RANKS = "ranks"
    NSFW_TAGS = "nsfw_tags"
    STUDIOS = "studios"
    USER_COMPLETE_DATE = "user_complete_date"
    START_SEASON = "start_season"
    RATING = "rating"
    FEATURES = "features"


class DefaultMapper:
    def __init__(self, name, default=pd.NA):
        self.name = name
        self.default = default

    def map(self, series):
        return series.get(self.name, self.default)


class SingleMapper:
    def __init__(self, name, func, default=pd.NA):
        self.name = name
        self.func = func
        self.default = default

    def map(self, dataframe):
        if self.name not in dataframe.columns and len(dataframe) > 0:
            dataframe[self.name] = self.default
            return dataframe[self.name]

        return dataframe[self.name].apply(self.row_wrapper, args=(self.func, self.default))

    def row_wrapper(self, row, func, default=None, args=None):
        args = args or []

        try:
            return func(row, *args)
        except (TypeError, ValueError, AttributeError, KeyError) as error:
            print(f"Error extracting {self.name}: {error}")
            return default


class MultiMapper:
    def __init__(self, func, default=pd.NA):
        self.func = func
        self.default = default

    def map(self, dataframe):
        return dataframe.apply(self.row_wrapper, args=(self.func, self.default), axis=1)

    def row_wrapper(self, row, func, default=None, args=None):
        args = args or []

        try:
            return func(row, *args)
        except (TypeError, ValueError, AttributeError, KeyError) as error:
            print(f"Error extracting with function {func}: {error}")
            return default
