import numpy as np

from animeippo.recommendation import scoring, util


class MostPopularCategory:
    description = "Most Popular for This Season"
    requires = [scoring.PopularityScorer.name]

    def categorize(self, dataset, max_items=10):
        target = dataset.recommendations

        return target.sort_values(scoring.PopularityScorer.name, ascending=False)[0:max_items]


class ContinueWatchingCategory:
    description = "Related to Your Completed Shows"
    requires = [scoring.ContinuationScorer.name]

    def categorize(self, dataset, max_items=None):
        target = dataset.recommendations

        return target[target[scoring.ContinuationScorer.name] > 0].sort_values(
            scoring.ContinuationScorer.name, ascending=False
        )[0:max_items]


class SourceCategory:
    description = "Based on a"
    requires = [scoring.SourceScorer.name, scoring.DirectSimilarityScorer.name]

    def categorize(self, dataset, max_items=10):
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

    def categorize(self, dataset, max_items=10):
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
