import numpy as np
import polars as pl
import datetime

from animeippo.recommendation import scoring, util, analysis, discourager


class MostPopularCategory:
    description = "Most Popular for This Year"
    requires = [scoring.PopularityScorer.name]

    def categorize(self, dataset, max_items=20):
        target = dataset.recommendations

        return target.sort(scoring.PopularityScorer.name, descending=True)[0:max_items]


class ContinueWatchingCategory:
    description = "Continue or Finish Watching"
    requires = [scoring.ContinuationScorer.name]

    def categorize(self, dataset, max_items=None):
        target = dataset.recommendations

        mask = (
            (pl.col(scoring.ContinuationScorer.name) > 0) & (pl.col("user_status") != "completed")
        ) | (pl.col("user_status") == "paused")

        return target.filter(mask).sort("final_score", descending=True)[0:max_items]


class AdaptationCategory:
    description = "Because You Read the Manga"
    requires = [scoring.AdaptationScorer]

    def categorize(self, dataset, max_items=None):
        target = dataset.recommendations

        return target.filter(
            (pl.col(scoring.AdaptationScorer.name) > 0) & (pl.col("user_status") != "completed")
        ).sort(scoring.AdaptationScorer.name, descending=True)[0:max_items]


# class SourceCategory:
#     description = "Based on a"
#     requires = [scoring.SourceScorer.name, scoring.DirectSimilarityScorer.name]

#     def categorize(self, dataset, max_items=20):
#         target = dataset.recommendations
#         compare = dataset.watchlist

#         source_mean = compare.groupby("source")["score"].mean()
#         weights = compare["source"].value_counts().sort("source")["count"].sqrt()
#         scores = weights * source_mean

#         best_source = pd.to_numeric(scores).idxmax(skipna=True)

#         if pd.isna(best_source) or len(best_source) <= 0:
#             best_source = "Manga"

#         target = target[(pd.isnull(target["user_status"]))]

#         match best_source.lower():
#             case "original":
#                 self.description = "Anime Originals"
#             case "other":
#                 self.description = "Unusual Sources"
#             case _:
#                 self.description = "Based on a " + str.title(best_source)

#         return target[target["source"] == best_source.lower()].sort("final_score", descending=True)[
#             0:max_items
#         ]


class StudioCategory:
    description = "From Your Favourite Studios"
    requires = [scoring.StudioCorrelationScorer.name, scoring.FormatScorer.name]

    def categorize(self, dataset, max_items=25):
        target = dataset.recommendations

        target = target.filter(
            (pl.col("user_status").is_null()) & (pl.col(scoring.FormatScorer.name) > 0.5)
        )

        return target.sort(
            [scoring.StudioCorrelationScorer.name, "final_score"], descending=[True, True]
        )[0:max_items]


# class ClusterCategory:
#     description = "X and Y Category"
#     requires = ["cluster"]

#     def __init__(self, nth_cluster):
#         self.nth_cluster = nth_cluster

#     def categorize(self, dataset, max_items=None):
#         target = dataset.recommendations
#         compare = dataset.watchlist

#         gdf = dataset.watchlist_explode_cached("features")

#         gdf = gdf.filter(~pl.col("features").is_in(dataset.nsfw_tags))

#         descriptions = util.extract_features(gdf["features"], gdf["cluster"], 2)

#         biggest_clusters = compare["cluster"].value_counts().index.to_list()

#         if self.nth_cluster < len(biggest_clusters):
#             cluster = biggest_clusters[self.nth_cluster]

#             mask = (pl.col("cluster") == cluster) & (pl.col("user_status").is_null())

#             relevant_shows = target[mask]

#             if len(relevant_shows) > 0:
#                 relevant = descriptions.iloc[cluster].tolist()

#                 self.description = " ".join(relevant)

#             return relevant_shows.sort("final_score", descending=True)[0:max_items]

#         return None


class GenreCategory:
    description = "Genre"
    requires = [scoring.GenreAverageScorer.name]

    def __init__(self, nth_genre):
        self.nth_genre = nth_genre

    def categorize(self, dataset, max_items=None):
        user_profile = dataset.user_profile

        if self.nth_genre < len(user_profile.genre_correlations):
            genre = user_profile.genre_correlations[self.nth_genre]["genres"].item()

            rdf = dataset.recommendations_explode_cached("genres")

            mask = (pl.col("genres") == genre) & (
                ~(pl.col("user_status").is_in(["completed", "dropped"]))
                | (pl.col("user_status").is_null())
            )

            relevant_shows = dataset.recommendations.filter(
                pl.col("id").is_in(rdf.filter(mask)["id"])
            )

            self.description = genre

            selected_shows = relevant_shows.sort("final_score", descending=True)[0:max_items]

            return selected_shows

        return None


class YourTopPicksCategory:
    description = "Top New Picks for You"
    requires = [scoring.ContinuationScorer.name]

    def categorize(self, dataset, max_items=25):
        target = dataset.recommendations

        mask = (
            (pl.col(scoring.ContinuationScorer.name) == scoring.ContinuationScorer.DEFAULT_SCORE)
            & (pl.col("user_status").is_null())
            & (pl.col("status").is_in(["releasing", "finished"]))
        )

        new_picks = target.filter(mask)

        return new_picks.sort("final_score", descending=True)[0:max_items]


class TopUpcomingCategory:
    description = "Top New Picks from Upcoming Anime"

    requires = [scoring.ContinuationScorer.name]

    def categorize(self, dataset, max_items=25):
        target = dataset.recommendations

        mask = (
            pl.col(scoring.ContinuationScorer.name) == scoring.ContinuationScorer.DEFAULT_SCORE
        ) & (pl.col("status") == "not_yet_released")

        new_picks = target.filter(mask)

        return new_picks.sort(by=["start_season", "final_score"], descending=[True, True])[
            0:max_items
        ]


class BecauseYouLikedCategory:
    description = "Because You Liked X"

    def __init__(self, nth_liked, distance_metric="jaccard"):
        self.nth_liked = nth_liked
        self.distance_metric = distance_metric

    def categorize(self, dataset, max_items=20):
        wl = dataset.watchlist
        target = dataset.recommendations

        target = target.filter(pl.col("user_status").is_null())

        mean = wl["score"].mean()

        mask = pl.col("score").ge(mean) & pl.col("user_complete_date").is_not_null()

        last_liked = wl.filter(mask).sort("user_complete_date", descending=True)

        if len(last_liked) > self.nth_liked:
            # We need a row, not an object
            liked_item = last_liked[self.nth_liked]

            self.description = "Because You Liked " + liked_item["title"].item()
            similarity = analysis.similarity_of_anime_lists(
                target["encoded"], liked_item["encoded"], self.distance_metric
            )
            return pl.concat(
                [target, pl.DataFrame(similarity.alias("similarity"))], how="horizontal"
            ).sort("similarity", descending=True)[0:max_items]

        return None


class SimulcastsCategory:
    description = "Top Simulcasts for You"

    def categorize(self, dataset, max_items=30):
        target = dataset.recommendations

        mask = pl.col("start_season") == self.get_current_season()
        simulcasts = target.filter(mask)

        return simulcasts.sort(by=["final_score"], descending=[True])[0:max_items]

    def get_current_season(self):
        today = datetime.date.today()

        season = ""

        if today.month in [1, 2, 3]:
            season = "winter"
        elif today.month in [4, 5, 6]:
            season = "spring"
        elif today.month in [7, 8, 9]:
            season = "summer"
        elif today.month in [10, 11, 12]:
            season = "fall"
        else:
            season = "?"

        yearseason = f"{today.year}/{season}"

        return yearseason


class PlanningCategory:
    description = "From Your Plan to Watch List"

    def categorize(self, dataset, max_items=30):
        target = dataset.recommendations

        mask = pl.col("user_status") == "planning"
        planning = target.filter(mask)

        return planning.sort(by=["final_score"], descending=[True])[0:max_items]


class DiscouragingWrapper:
    def __init__(self, category):
        self.category = category

    def categorize(self, dataset, **kwargs):
        dataset.recommendations = dataset.recommendations.with_columns(
            recommend_score=pl.col("recommend_score") * pl.col("discourage_score")
        )

        result = self.category.categorize(dataset, **kwargs)

        self.description = self.category.description

        # dataset.recommendations.loc[
        #     result.index, "discourage_score"
        # ] = discourager.apply_discourage_on_repeating_items(result["discourage_score"])

        # dataset.recommendations["final_score"] = dataset.recommendations["recommend_score"]

        return result


class DebugCategory:
    description = "Debug"

    def categorize(self, dataset, max_items=50):
        target = dataset.recommendations

        return target.sort("final_score", descending=True)[0:max_items]
