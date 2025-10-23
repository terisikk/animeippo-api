import polars as pl

from ..meta import meta
from . import scoring

FORMAT_ENUM = pl.Enum(
    [
        "TV",
        "MOVIE",
        "OVA",
        "ONA",
        "SPECIAL",
        "TV_SHORT",
        "MANGA",
        "ONE_SHOT",
        "NOVEL",
        "MUSIC",
    ]
)


class MostPopularCategory:
    description = "Most Popular for This Year"

    def categorize(self, dataset):
        mask = True
        sorting = {"by": scoring.PopularityScorer.name, "descending": True}

        return mask, sorting


class ContinueWatchingCategory:
    description = "Continue or Finish Watching"

    def categorize(self, dataset):
        mask = (
            (pl.col(scoring.ContinuationScorer.name) > 0)
            & (pl.col("user_status").ne_missing("completed"))
        ) | (pl.col("user_status") == "paused")

        # TODO: Cast format to enum already in the formatters
        by = [pl.col("format").cast(FORMAT_ENUM), "recommend_score"]
        descending = [False, True]

        sorting = {"by": by, "descending": descending}

        return mask, sorting


class AdaptationCategory:
    description = "Because You Read the Manga"

    def categorize(self, dataset):
        mask = (pl.col(scoring.AdaptationScorer.name) > 0) & (
            pl.col("user_status").ne_missing("completed")
        )

        by = [pl.col("format").cast(FORMAT_ENUM), scoring.AdaptationScorer.name]
        descending = [False, True]

        sorting = {"by": by, "descending": descending}

        return mask, sorting


class MangaCategory:
    description = "Based on a Manga"

    def categorize(self, dataset):
        mask = (
            (pl.col("user_status").is_null())
            & (pl.col("source") == "MANGA")
            & (pl.col("format").is_in(["TV", "MOVIE"]))
        )

        sorting = {"by": "recommend_score", "descending": True}

        return mask, sorting


class StudioCategory:
    description = "From Your Favourite Studios"

    def categorize(self, dataset):
        mask = pl.col("user_status").is_null()

        by = [
            scoring.StudioCorrelationScorer.name,
            pl.col("format").cast(FORMAT_ENUM),
            "recommend_score",
        ]
        descending = [True, False, True]

        return mask, {
            "by": by,
            "descending": descending,
        }


class GenreCategory:
    description = "Genre"

    def __init__(self, nth_genre=0):
        self.nth_genre = nth_genre

    def categorize(self, dataset):
        genre_correlations = dataset.user_profile.genre_correlations

        if self.nth_genre < len(genre_correlations):
            genre = genre_correlations[self.nth_genre]["name"].item()

            mask = (
                ~(pl.col("user_status").is_in(["completed", "dropped"]))
                | (pl.col("user_status").is_null())
            ) & (pl.col("genres").list.contains(genre))

            self.description = genre

            sorting = {"by": "recommend_score", "descending": True}

            return mask, sorting

        return False, {}


class YourTopPicksCategory:
    description = "Top New Picks for You"

    def categorize(self, dataset):
        mask = (
            (pl.col(scoring.ContinuationScorer.name) == 0)
            & (pl.col("user_status").is_null() | (pl.col("user_status") == "planning"))
            & (pl.col("status").is_in(["releasing", "finished"]))
        )

        sorting = {"by": "recommend_score", "descending": True}

        return mask, sorting


class TopUpcomingCategory:
    description = "Top Picks from Upcoming Anime"

    def categorize(self, dataset):
        year, season = meta.get_current_anime_season()

        mask = (pl.col("status") == "not_yet_released") & (
            (pl.col("season") > season) | (pl.col("season_year") > year)
        )

        sorting = {
            "by": ["season_year", "season", "recommend_score"],
            "descending": [False, False, True],
        }

        return mask, sorting


class BecauseYouLikedCategory:
    description = "Because You Liked X"

    def __init__(self, nth_liked, distance_metric="jaccard"):
        self.nth_liked = nth_liked
        self.distance_metric = distance_metric

    def categorize(self, dataset):
        last_liked = dataset.user_profile.last_liked

        if (
            last_liked is not None
            and len(last_liked) > self.nth_liked
            and dataset.similarity_matrix is not None
        ):
            liked_item = last_liked[self.nth_liked]

            try:
                similarity = dataset.get_similarity_matrix(filtered=False, transposed=True).select(
                    pl.col("id").cast(pl.UInt32),
                    pl.col(str(liked_item["id"].item())).alias("gscore"),
                )
            except pl.exceptions.ColumnNotFoundError:
                return False, {}

            self.description = f"Because You Liked {liked_item['title'].item()}"

            similar_anime = (
                dataset.recommendations.join(similarity, how="left", on="id")
                .filter(pl.col("user_status").is_null() & pl.col("gscore").is_not_nan())
                .sort(pl.col("gscore"), descending=True)
            )

            ids = similar_anime["id"].to_list()
            mask = pl.col("id").is_in(ids)

            # Sort by the id:s of similarity dataframe
            order_map = {id_: index for index, id_ in enumerate(ids)}
            sorting = {
                "by": pl.col("id").replace(order_map),
                "descending": False,
            }

            return mask, sorting

        return False, {}


class SimulcastsCategory:
    description = "Top Simulcasts for You"

    def categorize(self, dataset):
        year, season = meta.get_current_anime_season()
        mask = (pl.col("season_year") == year) & (pl.col("season") == season)
        sorting = {"by": "recommend_score", "descending": True}

        return mask, sorting


class PlanningCategory:
    description = "From Your Plan to Watch List"

    def categorize(self, dataset):
        mask = pl.col("user_status") == "planning"
        sorting = {"by": "recommend_score", "descending": True}

        return mask, sorting


class DebugCategory:
    description = "Debug"

    def categorize(self, dataset):
        mask = True  # Return all items
        sorting = {"by": "recommend_score", "descending": True}

        return mask, sorting
