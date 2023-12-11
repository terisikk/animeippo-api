import abc
import numpy as np
import pandas as pd

from animeippo.recommendation import analysis


class AbstractScorer(abc.ABC):
    @abc.abstractmethod
    def score(self, scoring_target_df, compare_df):
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass


class FeaturesSimilarityScorer(AbstractScorer):
    name = "featuresimilarityscore"

    def __init__(self, weighted=False, distance_metric=None):
        self.weighted = weighted
        self.distance_metric = distance_metric or "jaccard"

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        scores = analysis.similarity_of_anime_lists(
            scoring_target_df["encoded"],
            compare_df["encoded"].drop(scoring_target_df.index, errors="ignore"),
            self.distance_metric,
        )

        if self.weighted:
            averages = analysis.mean_score_per_categorical(
                compare_df.explode("features"), "features"
            )
            weights = scoring_target_df["features"].apply(
                analysis.weighted_mean_for_categorical_values, args=(averages.fillna(0.0),)
            )
            scores = scores * weights

        return analysis.normalize_column(scores)


class FeatureCorrelationScorer(AbstractScorer):
    name = "featurecorrelationscore"

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        score_correlations = analysis.weight_encoded_categoricals_correlation(
            compare_df, "encoded", data.all_features
        )

        dropped_or_paused = compare_df["user_status"].isin(["dropped", "paused"])

        drop_correlations = analysis.weight_encoded_categoricals_correlation(
            compare_df, "encoded", data.all_features, dropped_or_paused
        )

        scores = scoring_target_df["features"].apply(
            # For once, np.sum is the wanted metric,
            # so that titles with only a few features get lower scores
            # and titles with multiple good features get on top.
            # Slightly diminished effect with sqrt.
            analysis.weighted_sum_for_categorical_values,
            args=(score_correlations,),
        ) / np.sqrt(scoring_target_df["features"].str.len())

        scores = scores - (
            scoring_target_df["features"].apply(
                analysis.weighted_mean_for_categorical_values, args=(drop_correlations,)
            )
        )

        return analysis.normalize_column(scores)


class GenreAverageScorer(AbstractScorer):
    name = "genreaveragescore"

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        weights = analysis.weight_categoricals(compare_df, "genres")

        scores = scoring_target_df["genres"].apply(
            analysis.weighted_sum_for_categorical_values, args=(weights.fillna(0.0),)
        ) / np.sqrt(scoring_target_df["genres"].str.len())

        return analysis.normalize_column(scores)


# This gives way too much zero. Replace with mean / mode or just use the better averagescorer.
class StudioCountScorer(AbstractScorer):
    name = "studiocountscore"

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        counts = compare_df.explode("studios")["studios"].value_counts()

        scores = scoring_target_df.apply(self.max_studio_count, axis=1, args=(counts,))

        return analysis.normalize_column(scores)

    def max_studio_count(self, row, counts):
        if len(row["studios"]) == 0:
            return 0.0

        return np.max([counts.get(studio, 0.0) for studio in row["studios"]])


class StudioCorrelationScorer:
    name = "studiocorrelationscore"

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        weights = analysis.weight_categoricals_correlation(
            compare_df,
            "studios",
        )

        mode = weights.mode()

        mode = mode[0] if len(mode) > 0 else mode

        scores = scoring_target_df["studios"].apply(
            analysis.weighted_mean_for_categorical_values,
            args=(weights.fillna(mode),),
        )

        return analysis.normalize_column(scores)


class ClusterSimilarityScorer(AbstractScorer):
    name = "clusterscore"

    def __init__(self, weighted=False, distance_metric=None):
        self.weighted = weighted
        self.distance_metric = distance_metric or "jaccard"

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        st_encoded = np.vstack(scoring_target_df["encoded"])
        co_encoded = np.vstack(compare_df["encoded"])

        scores = pd.DataFrame(
            index=scoring_target_df.index, columns=range(0, len(compare_df["cluster"].unique()))
        )

        cluster_groups = compare_df.groupby("cluster")

        for cluster_id, cluster in cluster_groups:
            similarities = np.nanmean(
                analysis.similarity(
                    st_encoded,
                    co_encoded[cluster_groups.indices[cluster_id]],
                    metric=self.distance_metric,
                ),
                axis=1,
            )

            if self.weighted:
                averages = np.nanmean(cluster["score"].values)
                similarities = similarities * averages

            scores[cluster_id] = similarities

        if self.weighted:
            weights = np.sqrt(compare_df["cluster"].value_counts())
            scores = scores * weights

        return analysis.normalize_column(np.max(scores, axis=1))


class DirectSimilarityScorer(AbstractScorer):
    name = "directscore"

    def __init__(self, distance_metric=None):
        self.distance_metric = distance_metric or "jaccard"

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        similarities = analysis.categorical_similarity(
            scoring_target_df["encoded"],
            compare_df["encoded"].drop(scoring_target_df.index, errors="ignore"),
            metric=self.distance_metric,
        )

        max_columns = similarities.idxmax(axis=1).dropna().astype(int)
        scores = pd.Series(compare_df.loc[max_columns]["score"].values, index=max_columns.index)

        return analysis.normalize_column(scores)


class PopularityScorer(AbstractScorer):
    name = "popularityscore"

    def score(self, data):
        scoring_target_df = data.seasonal

        scores = scoring_target_df["popularity"]

        return analysis.normalize_column(scores.rank())


class ContinuationScorer(AbstractScorer):
    name = "continuationscore"

    DEFAULT_SCORE = 0
    DEFAULT_MEAN_SCORE = 5

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        mean_score = analysis.get_mean_score(compare_df, self.DEFAULT_MEAN_SCORE)

        rdf = scoring_target_df.explode("continuation_to")

        rdf["score"] = self.DEFAULT_SCORE

        is_continuation = rdf[rdf["continuation_to"].isin(compare_df.index)]
        sources = compare_df.loc[is_continuation["continuation_to"]]

        user_scores = sources["score"].fillna(mean_score)

        merged = pd.merge(rdf, user_scores, left_on="continuation_to", right_index=True)
        merged = self.get_max_score_of_duplicate_relations(merged, "score_y")

        rdf["score"].update(merged)

        scores = self.get_max_score_of_duplicate_relations(rdf, "score")

        return scores / 10

    def get_max_score_of_duplicate_relations(self, df, column):
        return df.groupby(df.index, sort=False)[column].max()


class AdaptationScorer(AbstractScorer):
    name = "adaptationscore"

    DEFAULT_SCORE = 0
    DEFAULT_MEAN_SCORE = 5

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.mangalist

        if compare_df is None:
            return np.nan

        mean_score = analysis.get_mean_score(compare_df, self.DEFAULT_MEAN_SCORE)

        rdf = scoring_target_df.explode("adaptation_of")
        rdf["score"] = self.DEFAULT_SCORE

        is_adaptation = rdf[rdf["adaptation_of"].isin(compare_df.index)]
        sources = compare_df.loc[is_adaptation["adaptation_of"]]

        user_scores = sources["score"].fillna(mean_score)

        merged = pd.merge(rdf, user_scores, left_on="adaptation_of", right_index=True)
        merged = self.get_max_score_of_duplicate_relations(merged, "score_y")

        rdf["score"].update(merged)

        scores = self.get_max_score_of_duplicate_relations(rdf, "score")

        return scores / 10

    def get_max_score_of_duplicate_relations(self, df, column):
        return df.groupby(df.index, sort=False)[column].max()


class SourceScorer(AbstractScorer):
    name = "sourcescore"

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        averages = compare_df.groupby("source")["score"].mean() / 10

        scores = scoring_target_df["source"].apply(lambda x: averages.get(x, 0))

        return analysis.normalize_column(scores)


class FormatScorer(AbstractScorer):
    name = "formatscore"

    FORMAT_MAPPING = {
        "TV": 1,
        "TV_SHORT": 0.8,
        "MOVIE": 1,
        "SPECIAL": 0.8,
        "OVA": 1,
        "ONA": 1,
        "MUSIC": 0.2,
        "MANGA": 1,
        "NOVEL": 1,
        "ONE_SHOT": 0.2,
    }

    def score(self, data):
        scoring_target_df = data.seasonal

        scores = scoring_target_df.apply(
            self.get_format_score,
            args=(
                scoring_target_df["episodes"].median(),
                scoring_target_df["duration"].median(),
            ),
            axis=1,
        )

        return analysis.normalize_column(scores)

    def get_format_score(self, row, median_episodes, median_duration):
        score = self.FORMAT_MAPPING.get(row["format"], 1)

        if row.get("episodes", median_episodes) < (0.75 * median_episodes) and row.get(
            "duration", median_duration
        ) < (0.75 * median_duration):
            score = score * 0.5

        return score


class DirectorCorrelationScorer:
    name = "directorcorrelationscore"

    def score(self, data):
        scoring_target_df = data.seasonal
        compare_df = data.watchlist

        weights = analysis.weight_categoricals_correlation(
            compare_df,
            "directors",
        )

        mode = weights.mode()

        mode = mode[0] if len(mode) > 0 else mode

        scores = scoring_target_df["directors"].apply(
            analysis.weighted_mean_for_categorical_values,
            args=(weights.fillna(mode),),
        )

        return analysis.normalize_column(scores)
