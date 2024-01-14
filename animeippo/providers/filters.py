import polars as pl


def MediaTypeFilter(*media_types, negative=False):
    """Filters a dataframe based on the media type (TV, Movie, etc.)
    field."""

    mask = pl.col("format").is_in(media_types)

    if negative:
        mask = ~mask

    return mask


def FeatureFilter(*features, negative=False):
    """Filters a dataframe based on feature names,
    for example genres or tags."""

    mask = pl.col("features").list.set_intersection(features) != []

    if negative:
        mask = ~mask

    return mask


def UserStatusFilter(*statuses, negative=False):
    """Filters a dataframe based on user status
    (completed, in_progress etc.) field."""

    mask = pl.col("user_status").is_in(statuses)

    if negative:
        mask = ~mask

    return mask


def RatingFilter(*ratings, negative=False):
    """Filters a dataframe based on rating (pg, r etc.) field."""

    mask = pl.col("rating").is_in(ratings)

    if negative:
        mask = ~mask

    return mask


def StartSeasonFilter(years, seasons=None, negative=False):
    """Filters a dataframe based on start season (2023/summer for example) fields."""

    if seasons is None or seasons == [None]:
        mask = pl.col("season_year").is_in(years)
    else:
        mask = pl.col("season_year").is_in(years) & pl.col("season").is_in(seasons)

    if negative:
        mask = ~mask

    return mask


def ContinuationFilter(compare_df, negative=False):
    """Filters a dataframe based on whether a series
    is a continuation or side story to a title that
    the user has already completed."""

    completed = compare_df.filter(pl.col("user_status") == "completed")["id"]

    mask = (pl.col("continuation_to").list.set_intersection(completed.to_list()) != []) | (
        pl.col("continuation_to") == []
    )

    if negative:
        mask = ~mask

    return mask
