import polars as pl

from ..meta import meta
from . import scoring


class MostPopularCategory:
    description = "Most Popular for This Year"

    def categorize(self, dataset):
        return True, {"by": scoring.PopularityScorer.name, "descending": True}


class ContinueWatchingCategory:
    description = "Continue or Finish Watching"

    def categorize(self, dataset):
        mask = (
            (pl.col(scoring.ContinuationScorer.name) > 0)
            & (pl.col("user_status").ne_missing("completed"))
        ) | (pl.col("user_status") == "paused")

        return mask, {"by": "recommend_score", "descending": True}


class AdaptationCategory:
    description = "Because You Read the Manga"

    def categorize(self, dataset):
        mask = (pl.col(scoring.AdaptationScorer.name) > 0) & (
            pl.col("user_status").ne_missing("completed")
        )

        return mask, {"by": scoring.AdaptationScorer.name, "descending": True}


class SourceCategory:
    description = "Based on a"

    def categorize(self, dataset):
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

        return mask, {"by": "adjusted_score", "descending": True}


class StudioCategory:
    description = "From Your Favourite Studios"

    FORMAT_THRESHOLD = 0.5

    def categorize(self, dataset):
        mask = (pl.col("user_status").is_null()) & (
            pl.col(scoring.FormatScorer.name) > self.FORMAT_THRESHOLD
        )

        return mask, {
            "by": [scoring.StudioCorrelationScorer.name, "recommend_score"],
            "descending": [True, True],
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

            return mask, {"by": "recommend_score", "descending": True}

        return False, {}


class YourTopPicksCategory:
    description = "Top New Picks for You"

    def categorize(self, dataset):
        mask = (
            (pl.col(scoring.ContinuationScorer.name) == 0)
            & (pl.col("user_status").is_null() | (pl.col("user_status") == "planning"))
            & (pl.col("status").is_in(["releasing", "finished"]))
        )

        return mask, {"by": "recommend_score", "descending": True}


class TopUpcomingCategory:
    description = "Top Picks from Upcoming Anime"

    def categorize(self, dataset):
        year, season = meta.get_current_anime_season()

        mask = (pl.col("status") == "not_yet_released") & (
            (pl.col("season") > season) | (pl.col("season_year") > year)
        )

        return mask, {
            "by": ["season_year", "season", "recommend_score"],
            "descending": [False, False, True],
        }


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
                similarity = []

            if len(similarity) > 0:
                self.description = f"Because You Liked {liked_item['title'].item()}"

                similar_anime = (
                    dataset.recommendations.join(similarity, how="left", on="id")
                    .filter(pl.col("user_status").is_null() & pl.col("gscore").is_not_nan())
                    .sort(pl.col("gscore"), descending=True)
                )

                ids = similar_anime["id"].to_list()
                mask = pl.col("id").is_in(ids)

                order_map = {id_: index for index, id_ in enumerate(ids)}

                # Sort by the id:s of similarity dataframe
                return mask, {
                    "by": pl.col("id").replace(order_map),
                    "descending": False,
                }

        return False, {}


class SimulcastsCategory:
    description = "Top Simulcasts for You"

    def categorize(self, dataset):
        year, season = meta.get_current_anime_season()
        mask = (pl.col("season_year") == year) & (pl.col("season") == season)

        return mask, {"by": "recommend_score", "descending": True}


class PlanningCategory:
    description = "From Your Plan to Watch List"

    def categorize(self, dataset):
        mask = pl.col("user_status") == "planning"

        return mask, {"by": "recommend_score", "descending": True}


class DebugCategory:
    description = "Debug"

    def categorize(self, dataset):
        mask = True

        return mask, {"by": "recommend_score", "descending": True}
