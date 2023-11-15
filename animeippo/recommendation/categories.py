import numpy as np
import pandas as pd
import datetime

from animeippo.recommendation import scoring, util, analysis


class MostPopularCategory:
    description = "Most Popular for This Year"
    requires = [scoring.PopularityScorer.name]

    def categorize(self, dataset, max_items=10):
        target = dataset.recommendations

        return target.sort_values(scoring.PopularityScorer.name, ascending=False)[0:max_items]


class ContinueWatchingCategory:
    description = "Related to Your Completed Shows"
    requires = [scoring.ContinuationScorer.name]

    def categorize(self, dataset, max_items=None):
        target = dataset.recommendations

        return target[
            (target[scoring.ContinuationScorer.name] > 0) & (target["user_status"] != "completed")
        ].sort_values(scoring.ContinuationScorer.name, ascending=False)[0:max_items]


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

        if "user_status" in target.columns:
            target = target[(pd.isnull(target["user_status"]))]

        match best_source.lower():
            case "original":
                self.description = "Anime Originals"
            case "other":
                self.description = "Unusual Sources"
            case _:
                self.description = "Based on a " + str.title(best_source)

        return target[target["source"] == best_source.lower()].sort_values(
            scoring.DirectSimilarityScorer.name, ascending=False
        )[0:max_items]


class StudioCategory:
    description = "From Your Favourite Studios"
    requires = [scoring.StudioCorrelationScorer.name]

    def categorize(self, dataset, max_items=20):
        target = dataset.recommendations

        target = target[(pd.isnull(target["user_status"]))]

        return target.sort_values(scoring.StudioCorrelationScorer.name, ascending=False)[
            0:max_items
        ]


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

            return relevant_shows[0:max_items]

        return None


class YourTopPicksCategory:
    description = "Top New Picks for You"
    requires = ["recommend_score", scoring.ContinuationScorer.name]

    def categorize(self, dataset, max_items=20):
        target = dataset.recommendations

        mask = (
            target[scoring.ContinuationScorer.name] == scoring.ContinuationScorer.DEFAULT_SCORE
        ) & (pd.isnull(target["user_status"]) & (target["status"].isin(["releasing", "finished"])))

        new_picks = target[mask]

        return new_picks.sort_values("recommend_score", ascending=False)[0:max_items]


class TopUpcomingCategory:
    description = "Top New Picks From Upcoming Anime"

    requires = ["recommend_score", scoring.ContinuationScorer.name]

    def categorize(self, dataset, max_items=20):
        target = dataset.recommendations

        mask = (
            target[scoring.ContinuationScorer.name] == scoring.ContinuationScorer.DEFAULT_SCORE
        ) & (target["status"] == "not_yet_released")

        new_picks = target[mask]

        return new_picks.sort_values(
            by=["start_season", "recommend_score"], ascending=[False, False]
        )[0:max_items]


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

            self.description = "Because You Liked " + last_liked.iloc[self.nth_liked]["title"]
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

        return simulcasts.sort_values(by=["recommend_score"], ascending=[False])[0:max_items]

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
