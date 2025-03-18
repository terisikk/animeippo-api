from enum import StrEnum

from polars import Enum


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
    TEMP_RANKS = "temp_ranks"
    STUDIOS = "studios"
    USER_COMPLETE_DATE = "user_complete_date"
    SEASON_YEAR = "season_year"
    SEASON = "season"
    RATING = "rating"
    FEATURES = "features"
    DIRECTOR = "directors"


Season = Enum(["winter", "spring", "summer", "fall"])
