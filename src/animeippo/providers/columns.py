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
    FRANCHISE = "franchise"
    FRANCHISE_RELATIONS = "franchise_relations"
    RECOMMENDATIONS = "recommendations"
    DIRECTOR = "directors"


Season = Enum(["WINTER", "SPRING", "SUMMER", "FALL"])
UserStatus = Enum(["CURRENT", "REPEATING", "COMPLETED", "DROPPED", "PAUSED", "PLANNING"])
MediaStatus = Enum(["FINISHED", "RELEASING", "NOT_YET_RELEASED", "CANCELLED", "HIATUS"])
MediaSource = Enum(
    [
        "ORIGINAL",
        "MANGA",
        "LIGHT_NOVEL",
        "VISUAL_NOVEL",
        "VIDEO_GAME",
        "OTHER",
        "NOVEL",
        "DOUJINSHI",
        "ANIME",
        "WEB_NOVEL",
        "LIVE_ACTION",
        "GAME",
        "COMIC",
        "MULTIMEDIA_PROJECT",
        "PICTURE_BOOK",
    ]
)
Format = Enum(
    [
        "TV",
        "TV_SHORT",
        "MOVIE",
        "SPECIAL",
        "OVA",
        "ONA",
        "MUSIC",
        "MANGA",
        "NOVEL",
        "ONE_SHOT",
    ]
)
