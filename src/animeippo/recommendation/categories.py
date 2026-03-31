import abc

import polars as pl

from ..meta import meta
from . import cluster_naming, scoring

WEAK_FORMATS = ["TV_SHORT", "SPECIAL", "MUSIC", "ONE_SHOT"]
MIN_EPISODE_DURATION = 10


def substantial_format_filter():
    """Filter out short-form and weak formats from premium lanes."""
    return ~pl.col("format").is_in(WEAK_FORMATS) & (
        pl.col("duration").fill_null(MIN_EPISODE_DURATION) >= MIN_EPISODE_DURATION
    )


class AbstractCategory(abc.ABC):
    def __init__(self, *, needs_diversity=False, min_items=1):
        self.needs_diversity = needs_diversity
        self.min_items = min_items

    @abc.abstractmethod
    def categorize(self, dataset):
        pass

    def get_items(self, df, top_n):
        return df["id"][0:top_n].to_list()


def compose_two_pool_lane(
    df,
    continuation_threshold=0.7,
    weak_interleave_interval=5,
    max_total=30,
    group_by=None,
):
    """Compose a lane by merging continuation and discovery pools.

    Strong continuations pin to the top of each group. Weak continuations
    interleave into the discovery list at regular intervals.

    When group_by is set, composition is applied per group (e.g. per season),
    preserving group order.
    """
    if group_by is not None:
        groups = df.select(group_by).unique(maintain_order=True).rows()
        result = []
        for group_vals in groups:
            group_filter = pl.lit(True)
            for col, val in zip(group_by, group_vals, strict=True):
                group_filter = group_filter & (pl.col(col) == val)
            group_df = df.filter(group_filter)
            result.extend(
                compose_single_pool(group_df, continuation_threshold, weak_interleave_interval)
            )
        if max_total is not None:
            result = result[:max_total]
        return result

    result = compose_single_pool(df, continuation_threshold, weak_interleave_interval)
    if max_total is not None:
        result = result[:max_total]
    return result


def compose_single_pool(df, continuation_threshold, weak_interleave_interval):
    """Compose a single pool: pin strong continuations, interleave weak ones."""
    cont_col = scoring.ContinuationScorer.name
    conf_col = f"{cont_col}_confidence"

    has_continuation = (pl.col(cont_col) > 0) & pl.col(conf_col).is_not_null()

    strong = (
        df.filter(has_continuation & (pl.col(conf_col) >= continuation_threshold))
        .sort(cont_col, descending=True)["id"]
        .to_list()
    )

    weak = (
        df.filter(has_continuation & (pl.col(conf_col) < continuation_threshold))
        .sort(cont_col, descending=True)["id"]
        .to_list()
    )

    excluded = strong + weak
    discoveries = df.filter(~pl.col("id").is_in(excluded))["id"].to_list()

    result = list(strong)

    interleave_idx = 0
    discovery_idx = 0
    slot = 1

    while discovery_idx < len(discoveries) or interleave_idx < len(weak):
        if interleave_idx < len(weak) and slot % weak_interleave_interval == 0:
            result.append(weak[interleave_idx])
            interleave_idx += 1
        elif discovery_idx < len(discoveries):
            result.append(discoveries[discovery_idx])
            discovery_idx += 1
        else:
            result.append(weak[interleave_idx])
            interleave_idx += 1
        slot += 1

    return result


class MostPopularCategory(AbstractCategory):
    description = "Most Popular for This Year"

    def categorize(self, dataset):
        mask = True
        sorting = {"by": "popularity", "descending": True}

        return mask, sorting


class ContinueWatchingCategory(AbstractCategory):
    description = "Continue or Finish Watching"

    def categorize(self, dataset):
        mask = (
            (pl.col(scoring.ContinuationScorer.name) > 0)
            & (pl.col("user_status").ne_missing("COMPLETED"))
        ) | (pl.col("user_status") == "PAUSED")

        by = [pl.col("format"), "discovery_score"]
        descending = [False, True]

        sorting = {"by": by, "descending": descending}

        return mask, sorting


class AdaptationCategory(AbstractCategory):
    description = "Because You Read the Manga"

    def categorize(self, dataset):
        mask = (pl.col(scoring.AdaptationScorer.name) > 0) & (
            pl.col("user_status").ne_missing("COMPLETED")
        )

        by = [pl.col("format"), scoring.AdaptationScorer.name]
        descending = [False, True]

        sorting = {"by": by, "descending": descending}

        return mask, sorting


class MangaCategory(AbstractCategory):
    description = "Based on a Manga"

    def categorize(self, dataset):
        mask = (
            (pl.col("user_status").is_null())
            & (pl.col("source") == "MANGA")
            & (pl.col("format").is_in(["TV", "MOVIE"]))
        )

        sorting = {"by": "discovery_score", "descending": True}

        return mask, sorting


class StudioCategory(AbstractCategory):
    description = "From Your Favourite Studios"

    def categorize(self, dataset):
        mask = pl.col("user_status").is_null()

        by = [
            scoring.StudioCorrelationScorer.name,
            pl.col("format"),
            "discovery_score",
        ]
        descending = [True, False, True]

        return mask, {
            "by": by,
            "descending": descending,
        }


class GenreCategory(AbstractCategory):
    description = "Genre"

    def __init__(self, nth_genre=0, **kwargs):
        super().__init__(**kwargs)
        self.nth_genre = nth_genre

    def categorize(self, dataset):
        genre_correlations = dataset.user_profile.genre_correlations

        if self.nth_genre < len(genre_correlations):
            genre = genre_correlations[self.nth_genre]["name"].item()

            mask = (
                ~(pl.col("user_status").is_in(["COMPLETED", "DROPPED"]))
                | (pl.col("user_status").is_null())
            ) & (pl.col("genres").list.contains(genre))

            self.description = genre

            sorting = {"by": "discovery_score", "descending": True}

            return mask, sorting

        return False, {}

    def get_items(self, df, top_n):
        return df["id"].to_list()


class TopReleasedPicksCategory(AbstractCategory):
    description = "Your Top 3"

    def categorize(self, dataset):
        mask = (
            (pl.col("status").is_in(["RELEASING", "FINISHED"]))
            & (pl.col("user_status").ne_missing("COMPLETED"))
            & substantial_format_filter()
        )

        sorting = {"by": "discovery_score", "descending": True}

        return mask, sorting


class HiddenGemsCategory(AbstractCategory):
    description = "Hidden Gems for You"

    def categorize(self, dataset):
        mask = (
            (pl.col("user_status").is_null() | (pl.col("user_status") == "PLANNING"))
            & (pl.col(scoring.ContinuationScorer.name) == 0)
            & (pl.col("status").is_in(["RELEASING", "FINISHED"]))
            & (pl.col("format").is_in(["TV", "OVA"]))
        )

        sorting = {
            "by": [
                pl.col("discovery_score")
                * (1 - 0.5 * pl.col("popularity").rank() / pl.col("popularity").count())
            ],
            "descending": True,
        }

        return mask, sorting


class MovieNightCategory(AbstractCategory):
    description = "Movie Night"

    def categorize(self, dataset):
        mask = (
            (pl.col("format") == "MOVIE")
            & (pl.col("user_status").ne_missing("COMPLETED"))
            & (pl.col("status").is_in(["RELEASING", "FINISHED"]))
        )

        sorting = {"by": "discovery_score", "descending": True}

        return mask, sorting


class AllMoviesCategory(AbstractCategory):
    description = "All Movies"

    def categorize(self, dataset):
        mask = (pl.col("format") == "MOVIE") & (pl.col("user_status").ne_missing("COMPLETED"))

        sorting = {"by": "discovery_score", "descending": True}

        return mask, sorting


class YourTopPicksCategory(AbstractCategory):
    description = "Top New Picks for You"

    def categorize(self, dataset):
        mask = (
            (pl.col(scoring.ContinuationScorer.name) == 0)
            & (pl.col("user_status").is_null() | (pl.col("user_status") == "PLANNING"))
            & (pl.col("status").is_in(["RELEASING", "FINISHED"]))
            & substantial_format_filter()
        )

        sorting = {"by": "discovery_score", "descending": True}

        return mask, sorting


class TopUpcomingCategory(AbstractCategory):
    description = "Top Picks from Upcoming Anime"

    CONTINUATION_THRESHOLD = 0.7
    WEAK_INTERLEAVE_INTERVAL = 5

    def categorize(self, dataset):
        year, season = meta.get_current_anime_season()

        mask = (
            (pl.col("status") == "NOT_YET_RELEASED")
            & ((pl.col("season") > season) | (pl.col("season_year") > year))
            & substantial_format_filter()
        )

        sorting = {
            "by": ["season_year", "season", "discovery_score"],
            "descending": [False, False, True],
        }

        return mask, sorting

    def get_items(self, df, top_n):
        return compose_two_pool_lane(
            df,
            continuation_threshold=self.CONTINUATION_THRESHOLD,
            weak_interleave_interval=self.WEAK_INTERLEAVE_INTERVAL,
            max_total=top_n,
            group_by=["season_year", "season"],
        )


class BecauseYouLikedCategory(AbstractCategory):
    description = "Because You Liked X"

    def __init__(self, nth_liked, distance_metric="cosine", **kwargs):
        super().__init__(**kwargs)
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


class SimulcastsCategory(AbstractCategory):
    description = "Top Simulcasts for You"

    CONTINUATION_THRESHOLD = 0.7
    WEAK_INTERLEAVE_INTERVAL = 5

    def categorize(self, dataset):
        year, season = meta.get_current_anime_season()
        mask = (
            (pl.col("season_year") == year)
            & (pl.col("season") == season)
            & substantial_format_filter()
        )
        sorting = {"by": "discovery_score", "descending": True}

        return mask, sorting

    def get_items(self, df, top_n):
        return compose_two_pool_lane(
            df,
            continuation_threshold=self.CONTINUATION_THRESHOLD,
            weak_interleave_interval=self.WEAK_INTERLEAVE_INTERVAL,
            max_total=top_n,
        )


class PlanningCategory(AbstractCategory):
    description = "From Your Plan to Watch List"

    def categorize(self, dataset):
        mask = pl.col("user_status") == "PLANNING"
        sorting = {
            "by": ["season_year", "season", "discovery_score"],
            "descending": [False, False, True],
        }

        return mask, sorting


class ClusterCategory(AbstractCategory):
    """Show seasonal recommendations matching a specific watchlist taste cluster."""

    description = "Cluster"

    def __init__(self, nth_cluster=0, tag_lookup=None, genres=None, **kwargs):
        super().__init__(**kwargs)
        self.nth_cluster = nth_cluster
        self.tag_lookup = tag_lookup or {}
        self.genres = genres or set()

    def categorize(self, dataset):
        watchlist = dataset.watchlist
        recommendations = dataset.recommendations

        if "cluster" not in watchlist.columns or "cluster" not in recommendations.columns:
            return False, {}

        ranked = cluster_naming.rank_clusters(watchlist, recommendations)

        if self.nth_cluster >= len(ranked):
            return False, {}

        cluster_id = ranked[self.nth_cluster]["cluster"]
        names = cluster_naming.name_all_clusters(watchlist, self.tag_lookup, self.genres)
        self.description = names.get(str(cluster_id), f"Cluster {cluster_id}")

        mask = pl.col("cluster") == cluster_id
        sorting = {"by": "discovery_score", "descending": True}

        return mask, sorting


class DebugCategory(AbstractCategory):
    description = "Debug"

    def categorize(self, dataset):
        mask = True  # Return all items
        sorting = {"by": "discovery_score", "descending": True}

        return mask, sorting
