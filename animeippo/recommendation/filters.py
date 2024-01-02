import abc
import polars as pl


class AbstractFilter(abc.ABC):
    @abc.abstractmethod
    def filter(self, dataset):
        pass


class MediaTypeFilter(AbstractFilter):
    """Filters a dataframe based on the media type (TV, Movie, etc.)
    field."""

    def __init__(self, *media_types, negative=False):
        self.media_types = media_types
        self.negative = negative

    def filter(self, dataframe):
        mask = pl.col("format").is_in(self.media_types)

        if self.negative:
            mask = ~mask

        return dataframe.filter(mask)


class FeatureFilter(AbstractFilter):
    """Filters a dataframe based on feature names,
    for example genres or tags."""

    def __init__(self, *features, negative=False):
        self.features = features
        self.negative = negative

    def filter(self, dataframe):
        mask = pl.col("features").list.set_intersection(self.features) != []

        if self.negative:
            mask = ~mask

        return dataframe.filter(mask)


class UserStatusFilter(AbstractFilter):
    """Filters a dataframe based on user status
    (completed, in_progress etc.) field."""

    def __init__(self, *statuses, negative=False):
        self.statuses = statuses
        self.negative = negative

    def filter(self, dataframe):
        mask = pl.col("user_status").is_in(self.statuses)

        if self.negative:
            mask = ~mask

        return dataframe.filter(mask)


class RatingFilter(AbstractFilter):
    """Filters a dataframe based on rating (pg, r etc.) field."""

    def __init__(self, *ratings, negative=False):
        self.ratings = ratings
        self.negative = negative

    def filter(self, dataframe):
        mask = dataframe["rating"].is_in(self.ratings)

        if self.negative:
            mask = ~mask

        return dataframe.filter(mask)


class StartSeasonFilter(AbstractFilter):
    """Filtres a dataframe based on start season (2023/summer for example) field."""

    def __init__(self, *seasons, negative=False):
        self.seasons = ["/".join(season) for season in seasons]
        self.negative = negative

    def filter(self, dataframe):
        mask = pl.col("start_season").is_in(self.seasons)

        if self.negative:
            mask = ~mask

        return dataframe.filter(mask)


class ContinuationFilter(AbstractFilter):
    """Filters a dataframe based on whether a series
    is a continuation or side story to a title that
    the user has already completed."""

    def __init__(self, compare_df, negative=False):
        self.compare_df = compare_df
        self.negative = negative

    def filter(self, dataframe):
        if dataframe["continuation_to"].dtype == pl.List(pl.Null):
            return dataframe

        completed = self.compare_df.filter(pl.col("user_status") == "completed")["id"]

        mask = (pl.col("continuation_to").list.set_intersection(completed.to_list()) != []) | (
            pl.col("continuation_to") == []
        )

        if self.negative:
            mask = ~mask

        return dataframe.filter(mask)
