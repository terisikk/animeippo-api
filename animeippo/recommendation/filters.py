import abc


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
        mask = dataframe["format"].isin(self.media_types)

        if self.negative:
            mask = ~mask

        return dataframe[mask]


class FeatureFilter(AbstractFilter):
    """Filters a dataframe based on feature names,
    for example genres or tags."""

    def __init__(self, *features, negative=False):
        self.features = features
        self.negative = negative

    def filter(self, dataframe):
        mask = dataframe["features"].apply(lambda field: all([feature in field for feature in self.features]))

        if self.negative:
            mask = ~mask

        return dataframe[mask]


class UserStatusFilter(AbstractFilter):
    """Filters a dataframe based on user status
    (completed, in_progress etc.) field."""

    def __init__(self, *statuses, negative=False):
        self.statuses = statuses
        self.negative = negative

    def filter(self, dataframe):
        mask = dataframe["user_status"].isin(self.statuses)

        if self.negative:
            mask = ~mask

        return dataframe[mask]


class IdFilter(AbstractFilter):
    """Filters a dataframe based on index labels."""

    def __init__(self, *ids, negative=False):
        self.ids = ids
        self.negative = negative

    def filter(self, dataframe):
        mask = dataframe.index.isin(self.ids)

        if self.negative:
            mask = ~mask

        return dataframe[mask]


class RatingFilter(AbstractFilter):
    """Filters a dataframe based on rating (pg, r etc.) field."""

    def __init__(self, *ratings, negative=False):
        self.ratings = ratings
        self.negative = negative

    def filter(self, dataframe):
        mask = dataframe["rating"].isin(self.ratings)

        if self.negative:
            mask = ~mask

        return dataframe[mask]


class StartSeasonFilter(AbstractFilter):
    """Filtres a dataframe based on start season (2023/summer for example) field."""

    def __init__(self, *seasons, negative=False):
        self.seasons = ["/".join(season) for season in seasons]
        self.negative = negative

    def filter(self, dataframe):
        mask = dataframe["start_season"].isin(self.seasons)

        if self.negative:
            mask = ~mask

        return dataframe[mask]


class ContinuationFilter(AbstractFilter):
    """Filters a dataframe based on whether a series
    is a continuation or side story to a title that
    the user has already completed."""

    def __init__(self, compare_df, negative=False):
        self.compare_df = compare_df
        self.negative = negative

    def filter(self, dataframe):
        completed = set(self.compare_df[self.compare_df["user_status"] == "completed"].index)

        mask = dataframe["continuation_to"].apply(self.filter_relations, args=(completed,))

        if self.negative:
            mask = ~mask

        return dataframe[mask]

    def filter_relations(self, item, completed):
        return bool(set(item) & completed) or len(item) == 0
