# TOOD: We could based on features + feature correlation
# (see feature correlation scorer and cluster namer and moods)
# select more interesting set of features than just genres
# to be shown on the frontend.
# Possible parts:
# * mood
# * most important genre
# * Setting (Time / Universe / Scene)
# * Theme
# * Demographic
# And from these somehow pick the most meaningful ones
# Maybe passing through categorical correlation, even though it's heavy.
# User profile could be cached better to redis to overcome.
# Well, recommendations too but it's no fun.
# Genre could even be discarded since theme mostly covers the same
# -> most important theme could be calculated from the same way as moods

from typing import ClassVar

import polars as pl

from animeippo.providers.anilist.data import TAG_BY_NAME


class AnimeDescriber:
    RANK_CUTOFF = 0.6

    DESCRIPTIVE_CATEGORIES = (
        "Theme",
        "Setting",
        "Cast",
        "Technical",
        "Genre",
    )

    # Not interesting from frontend point of view
    IGNORE_TAGS = (
        "Heterosexual",  # Or is it too heteronormative to leave it out?
        "Asexual",
        "Aromantic",
        "Bisexual",  # LGBTQ-themes will cover these better
        # A picture will say more
        "Primarily Male Cast",
        "Primarily Female Cast",
        "Primarily Adult Cast",
        "Primarily Child Cast",
        "Primarily Teen Cast",
        "Primarily Animal Cast",
        "Male Protagonist",
        "Female Protagonist",
        "Elderly Protagonist",
        "Ensemble Cast",
        # Does not say much about the show
        "Time Skip",
        "Achronological Order",
        "POV",
        "Advertisement",
        "CGI",  # Full CGI important, CGI not
        "Flash",
        "Achromatic",
    )

    GENRE_THEMES: ClassVar[dict[str, str]] = {
        "Action": "Theme-Action",
        "Comedy": "Theme-Comedy",
        "Drama": "Theme-Drama",
        "Fantasy": "Theme-Fantasy",
        "Romance": "Theme-Romance",
        "Sci-Fi": "Theme-Sci-Fi",
        "Slice of Life": "Theme-Slice of Life",
        "Music": "Theme-Arts-Music",
        "Sports": "Theme-Game-Sport",
    }

    # Lower = first
    OTHER_GENRE_PRIORITY: ClassVar[dict[str, int]] = {
        "Adventure": 1,
        "Ecchi": 0,
        "Hentai": 0,
        "Horror": 0,
        "Mahou Shoujo": 0,
        "Mecha": 0,
        "Mystery": 0,
        "Psychological": 0,
        "Supernatural": 0,
        "Thriller": 0,
    }

    ALWAYS_INCLUDE_CATEGORIES = ("Theme-Other",)

    def __init__(self, genres):
        self.tag_df = pl.from_dicts(list(TAG_BY_NAME.values()))

        self.genre_df = pl.DataFrame(
            list(genres.keys()),
            schema=["name"],
        )

    def fit(self):
        # dataset.all_features could be a dataframe instead of a list and contain
        # the categories etc directly.
        self.classified_features = pl.concat(
            [self.tag_df, self.genre_df.with_columns(category=pl.lit("Genre"))], how="diagonal"
        ).with_columns(
            pl.col("category").str.split("-").list.first().alias("category_short"),
        )

    def describe_shows(self, dataset, n_features=3):
        feature_correlations = dataset.user_profile.feature_correlations

        filtered_features = (
            dataset.recommendations.select(
                "id",
                "feature_info",
            )
            .explode("feature_info")
            .unnest("feature_info")
            .join(
                feature_correlations,
                on="name",
                how="left",
            )
            .join(
                self.classified_features.select("name", "category", "category_short"),
                on="name",
                how="left",
            )
            # FIXME: This does not work since genre rank is set to 100
            .with_columns(show_dynamic_cutoff=self.RANK_CUTOFF * pl.col("rank").max().over("id"))
            .filter(
                pl.col("rank") >= pl.col("show_dynamic_cutoff"),
                ~pl.col("name").is_in(self.IGNORE_TAGS),
                pl.col("category_short").is_in(self.DESCRIPTIVE_CATEGORIES),
                # TODO: Filter out spoiler tags, but needs new data to provider
                # ~pl.col("spoiler"),
            )
            .select(
                pl.col("id"),
                pl.col("name"),
                pl.col("weight"),
                pl.col("rank"),
                pl.col("category"),
                pl.col("category_short"),
            )
        )

        unique_features = filtered_features.select("id", "name").unique()

        substring_drops = (
            unique_features.join(
                unique_features.select(pl.col("id"), pl.col("name").alias("name_other")),
                on="id",
            )
            .filter(
                (pl.col("name") != pl.col("name_other"))
                & pl.col("name_other").str.contains(r"\b" + pl.col("name") + r"\b")
            )
            .select("id", "name")
            .unique()
        )

        descriptions = (
            filtered_features.join(
                substring_drops,
                on=["id", "name"],
                how="anti",
            )
            .with_columns(
                is_genre_theme=(pl.col("category").is_in(list(self.GENRE_THEMES.values())))
            )
            .with_columns(
                # Create a custom grouping bucket:
                # * Always include special cases
                # * Group all themes together,
                # * But for example Setting-* stays separate
                bucket=pl.when(pl.col("category").is_in(self.ALWAYS_INCLUDE_CATEGORIES))
                .then(pl.col("category"))
                .when(pl.col("category_short") == "Theme")
                .then(pl.lit("Theme"))
                .otherwise(pl.col("category_short"))
            )
            .with_columns(
                # Conditional sorting metric:
                # * Genre-themes get a boost (100) so they win
                # * Community rank to break ties among them.
                # * For everything else, fall back to user affinity weight
                sort_score=pl.when(pl.col("bucket") == "Theme")
                .then(pl.col("is_genre_theme").cast(pl.Float64) * 100 + pl.col("rank"))
                .otherwise(pl.col("weight"))
            )
            .sort(
                ["id", "bucket", "sort_score"],
                descending=[False, False, True],
            )
            .group_by(["id", "bucket"], maintain_order=True)
            .agg(pl.col("name").first().alias("description"))
            .group_by("id", maintain_order=True)
            .agg("description")
        )

        return descriptions
