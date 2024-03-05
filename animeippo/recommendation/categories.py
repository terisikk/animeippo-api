import polars as pl

from ..meta import meta
from . import scoring


class MostPopularCategory:
    description = "Most Popular for This Year"

    def categorize(self, dataset, max_items=20):
        return dataset.recommendations.sort(scoring.PopularityScorer.name, descending=True)[
            0:max_items
        ]


class ContinueWatchingCategory:
    description = "Continue or Finish Watching"

    def categorize(self, dataset, max_items=None):
        mask = (
            (pl.col(scoring.ContinuationScorer.name) > 0)
            & (pl.col("user_status").ne_missing("completed"))
        ) | (pl.col("user_status") == "paused")

        return dataset.recommendations.filter(mask).sort("final_score", descending=True)[
            0:max_items
        ]


class AdaptationCategory:
    description = "Because You Read the Manga"

    def categorize(self, dataset, max_items=None):
        mask = (pl.col(scoring.AdaptationScorer.name) > 0) & (
            pl.col("user_status").ne_missing("completed")
        )

        return dataset.recommendations.filter(mask).sort(
            scoring.AdaptationScorer.name, descending=True
        )[0:max_items]


class SourceCategory:
    description = "Based on a"

    def categorize(self, dataset, max_items=20):
        best_source = (
            dataset.watchlist.group_by("source")
            .agg(pl.col("score").mean() * pl.col("source").count().sqrt())
            .drop_nulls("score")
            .sort("score", descending=True)
            .select(pl.first("source"))
            .item()
        )

        if best_source is None:
            best_source = "Manga"

        mask = (pl.col("user_status").is_null()) & (pl.col("source") == best_source.lower())

        match best_source.lower():
            case "original":
                self.description = "Anime Originals"
            case "other":
                self.description = "Unusual Sources"
            case _:
                self.description = "Based on a " + str.title(best_source)

        return dataset.recommendations.filter(mask).sort("final_score", descending=True)[
            0:max_items
        ]


class StudioCategory:
    description = "From Your Favourite Studios"

    FORMAT_THRESHOLD = 0.5

    def categorize(self, dataset, max_items=25):
        mask = (pl.col("user_status").is_null()) & (
            pl.col(scoring.FormatScorer.name) > self.FORMAT_THRESHOLD
        )

        return dataset.recommendations.filter(mask).sort(
            [scoring.StudioCorrelationScorer.name, "final_score"], descending=[True, True]
        )[0:max_items]


class GenreCategory:
    description = "Genre"

    def __init__(self, nth_genre):
        self.nth_genre = nth_genre

    def categorize(self, dataset, max_items=None):
        genre_correlations = dataset.user_profile.genre_correlations

        if self.nth_genre < len(genre_correlations):
            genre = genre_correlations[self.nth_genre]["name"].item()

            mask = (pl.col("genres").list.contains(genre)) & (
                ~(pl.col("user_status").is_in(["completed", "dropped"]))
                | (pl.col("user_status").is_null())
            )

            self.description = genre

            return dataset.recommendations.filter(mask).sort("final_score", descending=True)[
                0:max_items
            ]

        return None


class YourTopPicksCategory:
    description = "Top New Picks for You"

    def categorize(self, dataset, max_items=25):
        mask = (
            (pl.col(scoring.ContinuationScorer.name) == scoring.ContinuationScorer.DEFAULT_SCORE)
            & (pl.col("user_status").is_null())
            & (pl.col("status").is_in(["releasing", "finished"]))
        )

        return dataset.recommendations.filter(mask).sort("final_score", descending=True)[
            0:max_items
        ]


class TopUpcomingCategory:
    description = "Top Picks from Upcoming Anime"

    def categorize(self, dataset, max_items=25):
        mask = (pl.col("status") == "not_yet_released") & (
            pl.col("season") > meta.get_current_anime_season()[1]
        )

        return dataset.recommendations.filter(mask).sort(
            by=["season_year", "season", "final_score"], descending=[False, False, True]
        )[0:max_items]


class BecauseYouLikedCategory:
    description = "Because You Liked X"

    def __init__(self, nth_liked, distance_metric="jaccard"):
        self.nth_liked = nth_liked
        self.distance_metric = distance_metric

    def categorize(self, dataset, max_items=20):
        last_liked = dataset.user_profile.last_liked

        if (
            last_liked is not None
            and len(last_liked) > self.nth_liked
            and dataset.similarity_matrix is not None
        ):
            liked_item = last_liked[self.nth_liked]

            try:
                similarity = dataset.get_similarity_matrix(filtered=False, transposed=True).select(
                    pl.col("id").cast(pl.Int64),
                    pl.col(str(liked_item["id"].item())).alias("gscore"),
                )
            except pl.ColumnNotFoundError:
                similarity = []

            if len(similarity) > 0:
                self.description = f"Because You Liked {liked_item['title'].item()}"

                return (
                    dataset.recommendations.join(similarity, how="left", on="id")
                    .filter(pl.col("user_status").is_null() & pl.col("gscore").is_not_nan())
                    .sort(pl.col("gscore"), descending=True)
                )[0:max_items]

        return None


class SimulcastsCategory:
    description = "Top Simulcasts for You"

    def categorize(self, dataset, max_items=30):
        year, season = meta.get_current_anime_season()
        mask = (pl.col("season_year") == year) & (pl.col("season") == season)

        return dataset.recommendations.filter(mask).sort(by=["final_score"], descending=[True])[
            0:max_items
        ]


class PlanningCategory:
    description = "From Your Plan to Watch List"

    def categorize(self, dataset, max_items=30):
        mask = pl.col("user_status") == "planning"

        return dataset.recommendations.filter(mask).sort(by=["final_score"], descending=[True])[
            0:max_items
        ]


class DiscouragingWrapper:
    DISCOURAGE_AMOUNT = 0.25

    def __init__(self, category):
        self.category = category

    def categorize(self, dataset, **kwargs):
        dataset.recommendations = dataset.recommendations.with_columns(
            final_score=pl.col("recommend_score") * pl.col("discourage_score")
        )

        result = self.category.categorize(dataset, **kwargs)

        self.description = self.category.description

        dataset.recommendations = dataset.recommendations.with_columns(
            discourage_score=pl.when(pl.col("id").is_in(result["id"]))
            .then(pl.col("discourage_score") - self.DISCOURAGE_AMOUNT)
            .otherwise(pl.col("discourage_score")),
        )

        return result


class DebugCategory:
    description = "Debug"

    def categorize(self, dataset, max_items=50):
        return dataset.recommendations.sort("final_score", descending=True)[0:max_items]
