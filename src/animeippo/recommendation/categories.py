import abc
from typing import ClassVar

import polars as pl

from animeippo.providers.anilist.data import ALL_GENRES

from ..meta import meta
from . import scoring

WEAK_FORMATS = ["TV_SHORT", "SPECIAL", "MUSIC", "ONE_SHOT"]
MIN_EPISODE_DURATION = 10

CONTINUATION_THRESHOLD = 0.7


def substantial_format_filter():
    """Filter out short-form and weak formats from premium lanes."""
    return ~pl.col("format").is_in(WEAK_FORMATS) & (
        pl.col("duration").fill_null(MIN_EPISODE_DURATION) >= MIN_EPISODE_DURATION
    )


def substantial_relation_filter():
    """Filter out for example summary/compilation movies from premium lanes."""
    return ~pl.col("is_summary")


class AbstractCategory(abc.ABC):
    def __init__(self, *, needs_diversity=False, exclusive=False, min_items=1):
        self.needs_diversity = needs_diversity
        self.exclusive = exclusive
        self.min_items = min_items

    @abc.abstractmethod
    def categorize(self, dataset):
        pass

    def get_items(self, df, top_n):
        return df["id"][0:top_n].to_list()


def merge_continuation_and_disovery(
    df,
    continuation_threshold=0.7,
    max_total=30,
    group_by=None,
):
    """Compose a lane by merging continuation and discovery pools.

    Strong continuations pin to the top of each group.

    When group_by is set, composition is applied per group (e.g. per season),
    preserving group order.
    """
    if group_by is not None:
        sub_dfs = df.partition_by(group_by, maintain_order=True, as_dict=False)
        result = []

        for sub_df in sub_dfs:
            group_ids = interleave_continuations(sub_df, continuation_threshold)
            result.extend(group_ids)

            if max_total is not None and len(result) >= max_total:
                result = result[:max_total]

        return result[:max_total] if max_total is not None else result

    # Flat feed, no grouping
    result = interleave_continuations(df, continuation_threshold)

    return result[:max_total] if max_total is not None else result


def interleave_continuations(df, continuation_threshold):
    """Compose a single pool: pin strong continuations"""
    cont_col = scoring.ContinuationScorer.name
    conf_col = f"{cont_col}_confidence"

    net_intent = pl.col(cont_col) * pl.col(conf_col)

    is_strong_continuation = pl.col(conf_col).is_not_null() & (net_intent >= continuation_threshold)

    # 1. Strong positive continuations pinned upfront, sorted by net intent.
    strong_ids = df.filter(is_strong_continuation).sort(net_intent, descending=True)["id"].to_list()

    # 2. All remaining items sorted by discovery score.
    discovery_ids = (
        df.filter(~is_strong_continuation).sort("discovery_score", descending=True)["id"].to_list()
    )

    return strong_ids + discovery_ids


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
            (
                (pl.col(scoring.ContinuationScorer.name) > 0)
                & (pl.col("user_status").ne_missing("COMPLETED"))
            )
            | (pl.col("user_status") == "PAUSED")
            | (pl.col("user_status") == "CURRENT")
        )

        sorting = {
            "by": pl.col("discovery_score")
            * pl.when(pl.col("is_summary")).then(0.3).otherwise(1.0),
            "descending": True,
        }

        return mask, sorting


class AdaptationCategory(AbstractCategory):
    description = "Because You Read the Manga"

    def categorize(self, dataset):
        mask = (pl.col(scoring.AdaptationScorer.name) > 0) & (
            pl.col("user_status").ne_missing("COMPLETED")
        )

        sorting = {"by": scoring.AdaptationScorer.name, "descending": True}

        return mask, sorting


class MangaCategory(AbstractCategory):
    description = "Based on a Manga"

    def categorize(self, dataset):
        mask = (
            (pl.col("user_status").is_null())
            & (pl.col("source") == "MANGA")
            & substantial_format_filter()
        )

        sorting = {"by": "discovery_score", "descending": True}

        return mask, sorting


class StudioCategory(AbstractCategory):
    """
    Pick top n studios and the best scoring item from them
    """

    description = "From Your Favourite Studios"

    def categorize(self, dataset):
        studio_correlations = dataset.user_profile.studio_correlations

        if studio_correlations is None or len(studio_correlations) == 0:
            return False, {}

        positive_studios = studio_correlations.filter(pl.col("weight") > 0)
        self.positive_studios = positive_studios

        if positive_studios.height == 0:
            return False, {}

        studio_list = positive_studios["name"].to_list()

        mask = (
            (
                ~(pl.col("user_status").is_in(["COMPLETED", "DROPPED"]))
                | (pl.col("user_status").is_null())
            )
            & (pl.col("studios").list.set_intersection(studio_list).list.len() > 0)
            & substantial_format_filter()
            & (pl.col("banner_image").is_not_null())
        )

        sorting = {"by": "discovery_score", "descending": True}

        return mask, sorting

    def get_items(self, df, top_n=None):
        """Sort by best matching studio weight, then discovery score."""
        result = (
            df.explode("studios")
            .join(self.positive_studios, left_on="studios", right_on="name", how="left")
            .group_by("id", maintain_order=True)
            .agg(
                pl.exclude("weight").first(),
                pl.col("weight").max(),
            )
            .sort("weight", "discovery_score", descending=True)
            .group_by("studios", maintain_order=True)
            .agg(pl.col("id").first())
        )

        if top_n is not None:
            result = result.head(top_n)

        return result["id"].to_list()


class GenreCategory(AbstractCategory):
    description = "Genre"

    def __init__(self, nth_genre=0, **kwargs):
        super().__init__(**kwargs)
        self.nth_genre = nth_genre

    def categorize(self, dataset):
        feature_correlations = dataset.user_profile.feature_correlations

        # FIXME: Really should not import all genres here,
        # should rather include category info and filter based on that
        genre_correlations = feature_correlations.filter(
            pl.col("name").is_in(ALL_GENRES.keys())
        ).sort("weight", descending=True)

        if genre_correlations is not None and self.nth_genre < genre_correlations.height:
            genre = genre_correlations[self.nth_genre]["name"].item()

            mask = (
                ~(pl.col("user_status").is_in(["COMPLETED", "DROPPED"]))
                | (pl.col("user_status").is_null())
            ) & (pl.col("genres").list.contains(genre))

            self.description = genre

            sorting = {"by": "discovery_score", "descending": True}

            return mask, sorting

        return False, {}

    def get_items(self, df, top_n=None):
        result = df

        if top_n is not None:
            result = result.head(top_n)

        return result["id"].to_list()


class TopReleasedPicksCategory(AbstractCategory):
    description = "Your Top 3"

    def categorize(self, dataset):
        mask = (
            (pl.col("status").is_in(["RELEASING", "FINISHED"]))
            & (pl.col("user_status").ne_missing("COMPLETED"))
            & substantial_format_filter()
            & substantial_relation_filter()
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
            & (pl.col("format") != "MOVIE")
            & (pl.col("banner_image").is_not_null())
            & substantial_format_filter()
            & substantial_relation_filter()
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
            & (pl.col("banner_image").is_not_null())
            & substantial_relation_filter()
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
            & (pl.col("banner_image").is_not_null())
            & substantial_format_filter()
        )

        sorting = {"by": "discovery_score", "descending": True}

        return mask, sorting


class TopUpcomingCategory(AbstractCategory):
    description = "Top Picks from Upcoming Anime"

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
        return merge_continuation_and_disovery(
            df,
            continuation_threshold=CONTINUATION_THRESHOLD,
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

            similarity = dataset.get_similarity_matrix(filtered=False, transposed=True).select(
                pl.col("id").cast(pl.UInt32),
                pl.col(str(liked_item["id"].item())).alias("gscore"),
            )

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
        return merge_continuation_and_disovery(
            df,
            continuation_threshold=CONTINUATION_THRESHOLD,
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

    def __init__(self, nth_cluster=0, **kwargs):
        super().__init__(**kwargs)
        self.nth_cluster = nth_cluster

    def categorize(self, dataset):
        clusters = dataset.clusters

        if clusters is None or self.nth_cluster >= len(clusters):
            return False, {}

        cluster = clusters.sort("rank_score", descending=True, nulls_last=True).row(
            self.nth_cluster, named=True
        )
        cluster_id = cluster.get("cluster", 0)
        cluster_name = cluster.get("name", "Unknown Cluster")

        self.description = cluster_name

        mask = pl.col("cluster") == cluster_id
        sorting = {
            "by": pl.col("discovery_score") * 0.5 + pl.col("cluster_similarity") * 0.5,
            "descending": True,
        }

        return mask, sorting


class MoodCategory(AbstractCategory):
    description = ""

    MOOD_SCORE_CUTOFF = 300

    MOOD_DESCRIPTIONS: ClassVar[dict[str, str]] = {
        "chill": "Relax & Unwind",
        "hype": "Get Hyped",
        "emotional": "Emotional Rollercoasters",
        "dark": "Grim & Gritty Tales",
        "funny": "Goofy Comedies",
        "cerebral": "Mind-Bending Mysteries",
        "adventurous": "Embark on Grand Journeys",
        "sporty": "Sweat it Out",
    }

    def categorize(self, dataset):
        mood_correlations = dataset.user_profile.mood_correlations

        if mood_correlations is None or mood_correlations.is_empty():
            return False, {}

        top_mood = mood_correlations[0, "name"]
        mask = (
            (
                pl.col("moods")
                .list.eval(
                    (pl.element().struct.field("mood") == top_mood)
                    & (pl.element().struct.field("score") >= self.MOOD_SCORE_CUTOFF)
                )
                .list.any()
            )
            & (
                ~(pl.col("user_status").is_in(["COMPLETED", "DROPPED"]))
                | (pl.col("user_status").is_null())
            )
            & substantial_format_filter()
            & pl.col("banner_image").is_not_null()
        )

        # Or we could instead sort by mood score and select the best?
        sorting = {"by": "discovery_score", "descending": True}

        self.description = self.MOOD_DESCRIPTIONS.get(top_mood, top_mood.capitalize())

        return mask, sorting


class DebugCategory(AbstractCategory):
    description = "Debug"

    def categorize(self, dataset):
        mask = True  # Return all items
        sorting = {"by": "discovery_score", "descending": True}

        return mask, sorting
