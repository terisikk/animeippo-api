import numpy as np
import pandas as pd

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

        best_source = scores.idxmax()

        match best_source.lower():
            case "original":
                self.description = "Anime Originals"
            case "other":
                self.description = "Unusual Sources"
            case _:
                self.description = "Based on a " + str.title(best_source)

        return target[target["source"] == best_source].sort_values(
            scoring.DirectSimilarityScorer.name, ascending=False
        )[0:max_items]


class StudioCategory:
    description = "From Your Favourite Studios"
    requires = [scoring.StudioAverageScorer.name]

    def categorize(self, dataset, max_items=20):
        target = dataset.recommendations

        return target.sort_values(scoring.StudioAverageScorer.name, ascending=False)[0:max_items]


class ClusterCategory:
    description = "X and Y Category"
    requires = ["cluster"]

    def __init__(self, nth_cluster):
        self.nth_cluster = nth_cluster

    def categorize(self, dataset, max_items=None):
        target = dataset.recommendations
        compare = dataset.watchlist

        gdf = compare.explode("features")

        descriptions = util.extract_features(gdf["features"], gdf["cluster"], 2)

        biggest_clusters = compare["cluster"].value_counts().index.to_list()

        if self.nth_cluster < len(biggest_clusters):
            cluster = biggest_clusters[self.nth_cluster]

            relevant_shows = target[target["cluster"] == cluster]

            if len(relevant_shows) > 0:
                relevant = descriptions.iloc[cluster].tolist()

                self.description = " ".join(relevant)

            return relevant_shows[0:max_items]

        return None


class YourTopPicks:
    description = "Top New Picks for You"
    requires = ["recommend_score", scoring.ContinuationScorer.name]

    def categorize(self, dataset, max_items=20):
        target = dataset.recommendations

        mask = (
            target[scoring.ContinuationScorer.name] == scoring.ContinuationScorer.DEFAULT_SCORE
        ) & (pd.isnull(target["user_status"]) & (target["status"].isin(["releasing", "finished"])))

        new_picks = target[mask]

        return new_picks.sort_values("recommend_score", ascending=False)[0:max_items]


class TopUpcoming:
    description = "Top New Picks From Upcoming Anime"

    requires = ["recommend_score", scoring.ContinuationScorer.name]

    def categorize(self, dataset, max_items=20):
        target = dataset.recommendations

        mask = (
            target[scoring.ContinuationScorer.name] == scoring.ContinuationScorer.DEFAULT_SCORE
        ) & (target["status"] == "not_yet_released")

        new_picks = target[mask]

        return new_picks.sort_values("recommend_score", ascending=False)[0:max_items]


class BecauseYouLiked:
    description = "Because You Liked X"

    def __init__(self, nth_liked):
        self.nth_liked = nth_liked

    def categorize(self, dataset, max_items=20):
        wl = dataset.watchlist

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
                dataset.recommendations["encoded"], liked_item["encoded"]
            )
            return similarity.sort_values(ascending=False)[0:max_items]

        return None
