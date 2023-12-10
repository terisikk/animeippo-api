import numpy as np
import pandas as pd
import datetime

from animeippo.recommendation import scoring, util, analysis, discourager


class MostPopularCategory:
    description = "Most Popular for This Year"
    requires = [scoring.PopularityScorer.name]

    def categorize(self, dataset, max_items=20):
        target = dataset.recommendations

        return target.sort_values(scoring.PopularityScorer.name, ascending=False)[0:max_items]


class ContinueWatchingCategory:
    description = "Related to Your Completed Shows"
    requires = [scoring.ContinuationScorer.name]

    def categorize(self, dataset, max_items=None):
        target = dataset.recommendations

        return target[
            (target[scoring.ContinuationScorer.name] > 0) & (target["user_status"] != "completed")
        ].sort_values("final_score", ascending=False)[0:max_items]


class AdaptationCategory:
    description = "Because You Read the Manga"
    requires = [scoring.AdaptationScorer]

    def categorize(self, dataset, max_items=None):
        target = dataset.recommendations

        return target[
            (target[scoring.AdaptationScorer.name] > 0) & (target["user_status"] != "completed")
        ].sort_values(scoring.AdaptationScorer.name, ascending=False)[0:max_items]


class SourceCategory:
    description = "Based on a"
    requires = [scoring.SourceScorer.name, scoring.DirectSimilarityScorer.name]

    def categorize(self, dataset, max_items=20):
        target = dataset.recommendations
        compare = dataset.watchlist

        source_mean = compare.groupby("source")["score"].mean()
        weights = np.sqrt(compare["source"].value_counts())
        scores = weights * source_mean

        best_source = pd.to_numeric(scores).idxmax(skipna=True)

        if pd.isna(best_source) or len(best_source) <= 0:
            best_source = "Manga"

        target = target[(pd.isnull(target["user_status"]))]

        match best_source.lower():
            case "original":
                self.description = "Anime Originals"
            case "other":
                self.description = "Unusual Sources"
            case _:
                self.description = "Based on a " + str.title(best_source)

        return target[target["source"] == best_source.lower()].sort_values(
            "final_score", ascending=False
        )[0:max_items]


class StudioCategory:
    description = "From Your Favourite Studios"
    requires = [scoring.StudioCorrelationScorer.name, scoring.FormatScorer.name]

    def categorize(self, dataset, max_items=25):
        target = dataset.recommendations

        target = target[
            (pd.isnull(target["user_status"])) & (target[scoring.FormatScorer.name] > 0.5)
        ]

        return target.sort_values(
            [scoring.StudioCorrelationScorer.name, "final_score"], ascending=[False, False]
        )[0:max_items]


class ClusterCategory:
    description = "X and Y Category"
    requires = ["cluster"]

    def __init__(self, nth_cluster):
        self.nth_cluster = nth_cluster

    def categorize(self, dataset, max_items=None):
        target = dataset.recommendations
        compare = dataset.watchlist

        gdf = compare.explode("features")

        gdf = gdf[~gdf["features"].isin(dataset.nsfw_tags)]

        descriptions = util.extract_features(gdf["features"], gdf["cluster"], 2)

        biggest_clusters = compare["cluster"].value_counts().index.to_list()

        if self.nth_cluster < len(biggest_clusters):
            cluster = biggest_clusters[self.nth_cluster]

            mask = (target["cluster"] == cluster) & (pd.isnull(target["user_status"]))

            relevant_shows = target[mask]

            if len(relevant_shows) > 0:
                relevant = descriptions.iloc[cluster].tolist()

                self.description = " ".join(relevant)

            return relevant_shows.sort_values("final_score", ascending=False)[0:max_items]

        return None


class GenreCategory:
    description = "Genre"
    requires = [scoring.GenreAverageScorer.name]

    def __init__(self, nth_genre):
        self.nth_genre = nth_genre

    def categorize(self, dataset, max_items=None):
        if dataset.user_favourite_genres is None:
            gdf = dataset.watchlist_exploded_by_genres

            gdf = gdf[~gdf["genres"].isin(dataset.nsfw_tags)]

            dataset.user_favourite_genres = analysis.weight_categoricals_correlation(
                gdf, "genres"
            ).sort_values(ascending=False)

        if self.nth_genre < len(dataset.user_favourite_genres):
            genre = dataset.user_favourite_genres.index[self.nth_genre]

            tdf = dataset.recommendations_exploded_by_genres

            mask = (tdf["genres"] == genre) & ~(tdf["user_status"].isin(["completed", "dropped"]))

            relevant_shows = tdf[mask]

            self.description = genre

            selected_shows = relevant_shows.sort_values("final_score", ascending=False)[0:max_items]

            return selected_shows

        return None


class YourTopPicksCategory:
    description = "Top New Picks for You"
    requires = [scoring.ContinuationScorer.name]

    def categorize(self, dataset, max_items=25):
        target = dataset.recommendations

        mask = (
            target[scoring.ContinuationScorer.name] == scoring.ContinuationScorer.DEFAULT_SCORE
        ) & (pd.isnull(target["user_status"]) & (target["status"].isin(["releasing", "finished"])))

        new_picks = target[mask]

        return new_picks.sort_values("final_score", ascending=False)[0:max_items]


class TopUpcomingCategory:
    description = "Top New Picks from Upcoming Anime"

    requires = [scoring.ContinuationScorer.name]

    def categorize(self, dataset, max_items=25):
        target = dataset.recommendations

        mask = (
            target[scoring.ContinuationScorer.name] == scoring.ContinuationScorer.DEFAULT_SCORE
        ) & (target["status"] == "not_yet_released")

        new_picks = target[mask]

        return new_picks.sort_values(by=["start_season", "final_score"], ascending=[False, False])[
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

        target = target[(pd.isnull(target["user_status"]))]

        mean = wl["score"].mean()

        last_complete = wl[pd.notna(wl["user_complete_date"])].sort_values(
            "user_complete_date", ascending=False
        )

        last_liked = last_complete[last_complete["score"].ge(mean)]

        if len(last_liked) > self.nth_liked:
            # We need a row, not an object
            liked_item = last_liked.iloc[self.nth_liked : self.nth_liked + 1]

            self.description = "Because You Liked " + liked_item["title"].iloc[0]
            similarity = analysis.similarity_of_anime_lists(
                target["encoded"], liked_item["encoded"], self.distance_metric
            )
            return similarity.sort_values(ascending=False)[0:max_items]

        return None


class SimulcastsCategory:
    description = "Top Simulcasts for You"

    def categorize(self, dataset, max_items=30):
        target = dataset.recommendations

        mask = target["start_season"] == self.get_current_season()
        simulcasts = target[mask]

        return simulcasts.sort_values(by=["final_score"], ascending=[False])[0:max_items]

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


class DiscouragingWrapper:
    def __init__(self, category):
        self.category = category

    def categorize(self, dataset, **kwargs):
        dataset.recommendations["final_score"] = (
            dataset.recommendations["recommend_score"] * dataset.recommendations["discourage_score"]
        )

        result = self.category.categorize(dataset, **kwargs)

        self.description = self.category.description

        dataset.recommendations.loc[
            result.index, "discourage_score"
        ] = discourager.apply_discourage_on_repeating_items(result)

        dataset.recommendations["final_score"] = dataset.recommendations["recommend_score"]

        return result


class DebugCategory:
    description = "Debug"

    def categorize(self, dataset, max_items=50):
        target = dataset.recommendations

        return target.sort_values("final_score", ascending=False)[0:max_items]
