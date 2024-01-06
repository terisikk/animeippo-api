import polars as pl
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
    DIRECTOR = "directors"


class DefaultMapper:
    def __init__(self, name, default=None):
        self.name = name
        self.default = default

    def map(self, series):
        return series.get_column(self.name) if self.name in series.columns else pl.lit(self.default)


class SelectorMapper:
    def __init__(self, selector):
        self.selector = selector

    def map(self, dataframe):
        return dataframe.select(self.selector).to_series()


class QueryMapper:
    def __init__(self, query):
        self.query = query

    def map(self, dataframe):
        return self.query(dataframe)


class SingleMapper:
    def __init__(self, name, func, default=None):
        self.name = name
        self.func = func
        self.default = default

    def map(self, dataframe):
        if self.name not in dataframe.columns and len(dataframe) > 0:
            return pl.lit(self.default)

        return dataframe[self.name].map_elements(self.row_wrapper)

    def row_wrapper(self, row):
        try:
            return self.func(row)
        except (TypeError, ValueError, AttributeError, KeyError) as error:
            print(f"Error extracting {self.name}: {error}")
            return self.default


class MultiMapper:
    def __init__(self, columns, func, default=None):
        self.columns = columns
        self.func = func
        self.default = default

    def map(self, dataframe):
        if any((column not in dataframe.columns for column in self.columns)):
            return pl.lit(self.default)

        return dataframe.select(self.columns).map_rows(self.row_wrapper).to_series()

    def row_wrapper(self, row):
        try:
            return self.func(*row)
        except (TypeError, ValueError, AttributeError, KeyError) as error:
            print(f"Error extracting with function {self.func}: {error}")
            return self.default
