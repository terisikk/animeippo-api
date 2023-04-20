import abc


class AbstractFilter(abc.ABC):
    @abc.abstractmethod
    def filter(self, dataframe):
        pass


class MediaTypeFilter(AbstractFilter):
    def __init__(self, *media_types, negative=False):
        self.media_types = media_types
        self.negative = negative

    def filter(self, dataframe):
        mask = dataframe["media_type"].isin(self.media_types)

        if self.negative:
            mask = ~mask

        return dataframe[mask]


class GenreFilter(AbstractFilter):
    def __init__(self, *genres, negative=False):
        self.genres = genres
        self.negative = negative

    def filter(self, dataframe):
        mask = dataframe["genres"].apply(
            lambda genres: all([genre in genres for genre in self.genres])
        )

        if self.negative:
            mask = ~mask

        return dataframe[mask]


class StatusFilter(AbstractFilter):
    def __init__(self, *statuses, negative=False):
        self.statuses = statuses
        self.negative = negative

    def filter(self, dataframe):
        mask = dataframe["status"].isin(self.statuses)

        if self.negative:
            mask = ~mask

        return dataframe[mask]


class IdFilter(AbstractFilter):
    def __init__(self, *ids, negative=False):
        self.ids = ids
        self.negative = negative

    def filter(self, dataframe):
        mask = dataframe.index.isin(self.ids)

        if self.negative:
            mask = ~mask

        return dataframe[mask]


class RatingFilter(AbstractFilter):
    def __init__(self, *ratings, negative=False):
        self.ratings = ratings
        self.negative = negative

    def filter(self, dataframe):
        mask = dataframe["rating"].isin(self.ratings)

        if self.negative:
            mask = ~mask

        return dataframe[mask]


class StartSeasonFilter(AbstractFilter):
    def __init__(self, *seasons, negative=False):
        self.seasons = ["/".join(season) for season in seasons]
        self.negative = negative

    def filter(self, dataframe):
        mask = dataframe["start_season"].isin(self.seasons)

        if self.negative:
            mask = ~mask

        return dataframe[mask]


class ContinuationFilter(AbstractFilter):
    def __init__(self, compare_df, negative=False):
        self.compare_df = compare_df
        self.negative = negative

    def filter(self, dataframe):
        ids = self.compare_df.index.values

        mask = dataframe.apply(
            lambda row: any([i in row["related_anime"] for i in ids])
            or len(row["related_anime"]) == 0,
            axis=1,
        )

        if self.negative:
            mask = ~mask

        return dataframe[mask]
