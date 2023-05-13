import abc
import numpy as np
import pandas as pd
import sklearn.preprocessing as skpre
import sklearn.cluster as skcluster

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

    def __init__(self, weighted=False):
        self.weighted = weighted

    def score(self, scoring_target_df, compare_df):
        all_features = (
            pd.concat([compare_df["features"], scoring_target_df["features"]]).explode().unique()
        )

        encoder = CategoricalEncoder(all_features)

        scores = analysis.similarity_of_anime_lists(
            scoring_target_df["features"], compare_df["features"], encoder
        )

        if self.weighted:
            averages = analysis.mean_score_per_categorical(compare_df, "features")
            weights = scoring_target_df["features"].apply(
                analysis.weighted_mean_for_categorical_values, args=(averages,)
            )
            scores = scores * weights

        return analysis.normalize_column(scores)


class GenreAverageScorer(AbstractScorer):
    name = "genreaveragescore"

    def score(self, scoring_target_df, compare_df):
        weights = analysis.weight_categoricals(compare_df, "genres")

        scores = scoring_target_df["genres"].apply(
            analysis.weighted_mean_for_categorical_values, args=(weights,)
        )

        return analysis.normalize_column(scores)


# This gives way too much zero. Replace with mean / mode or just use the better averagescorer.
class StudioCountScorer(AbstractScorer):
    name = "studiocountscore"

    def score(self, scoring_target_df, compare_df):
        counts = compare_df.explode("studios")["studios"].value_counts()

        scores = scoring_target_df.apply(self.max_studio_count, axis=1, args=(counts,))

        return analysis.normalize_column(scores)

    def max_studio_count(self, row, counts):
        if len(row["studios"]) == 0:
            return 0.0

        return np.max([counts.get(studio, 0.0) for studio in row["studios"]])


class StudioAverageScorer:
    name = "studioaveragescore"

    def score(self, scoring_target_df, compare_df):
        weights = analysis.weight_categoricals(compare_df, "studios")

        scores = scoring_target_df["studios"].apply(
            analysis.weighted_mean_for_categorical_values, args=(weights, weights.mode()[0])
        )

        return analysis.normalize_column(scores)


class ClusterSimilarityScorer(AbstractScorer):
    name = "clusterscore"

    def __init__(self, weighted=False):
        self.model = skcluster.AgglomerativeClustering(
            n_clusters=None, metric="precomputed", linkage="average", distance_threshold=0.7
        )

        self.weighted = weighted

    def score(self, scoring_target_df, compare_df):
        scores = pd.DataFrame(index=scoring_target_df.index)

        all_features = (
            pd.concat([compare_df["features"], scoring_target_df["features"]]).explode().unique()
        )

        encoder = CategoricalEncoder(all_features)

        encoded = encoder.encode(compare_df["features"])
        distances = pd.DataFrame(1 - analysis.similarity(encoded, encoded), index=compare_df.index)

        compare_df["cluster"] = self.model.fit_predict(distances)

        for cluster_id, cluster in compare_df.groupby("cluster"):
            similarities = analysis.similarity_of_anime_lists(
                scoring_target_df["features"], cluster["features"], encoder
            )

            if self.weighted:
                averages = cluster["score"].mean()
                similarities = similarities * averages

            scores[cluster_id] = similarities

        if self.weighted:
            weights = np.sqrt(compare_df["cluster"].value_counts())
            scores = scores * weights

        return analysis.normalize_column(scores.apply(np.max, axis=1))


class DirectSimilarityScorer(AbstractScorer):
    name = "directscore"

    def score(self, scoring_target_df, compare_df):
        all_features = (
            pd.concat([compare_df["features"], scoring_target_df["features"]]).explode().unique()
        )

        encoder = CategoricalEncoder(all_features)

        similarities = pd.DataFrame(
            analysis.similarity(
                encoder.encode(scoring_target_df["features"]),
                encoder.encode(compare_df["features"]),
            ),
            index=scoring_target_df.index,
        )

        max_values = similarities.max(axis=1)
        max_columns = similarities[similarities.eq(max_values, axis=0)]

        max_columns_list = max_columns.notna().apply(
            lambda row: list(max_columns.columns[row]), axis=1
        )

        scores = (
            max_values
            * max_columns_list.apply(lambda x: compare_df.iloc[x]["score"].mean()).fillna(6.0)
            / 10
        )

        return analysis.normalize_column(scores)


class PopularityScorer(AbstractScorer):
    name = "popularityscore"

    def score(self, scoring_target_df, compare_df):
        scores = scoring_target_df["popularity"]

        return analysis.normalize_column(scores.rank())


class ContinuationScorer(AbstractScorer):
    name = "continuationscore"

    SCALE_MIDDLE = 6

    def score(self, scoring_target_df, compare_df):
        mean_score = compare_df["score"].mean()

        rdf = scoring_target_df.explode("related_anime")

        rdf["score"] = np.nan

        for i, row in rdf.iterrows():
            related_index = row["related_anime"]
            is_continuation = related_index in compare_df.index
            rdf.at[i, "score"] = (
                compare_df.at[related_index, "score"] if is_continuation else self.SCALE_MIDDLE
            )

        rdf["score"] = rdf["score"].fillna(mean_score)

        scores = rdf.groupby("id")["score"].max()

        return scores / 10


class SourceScorer(AbstractScorer):
    name = "sourcescore"

    def score(self, scoring_target_df, compare_df):
        averages = compare_df.groupby("source")["score"].mean() / 10

        scores = scoring_target_df["source"].apply(lambda x: averages.get(x, 0))

        return analysis.normalize_column(scores)


class CategoricalEncoder:
    def __init__(self, classes):
        self.classes = classes
        self.mlb = skpre.MultiLabelBinarizer(classes=classes)
        self.mlb.fit(None)

    def encode(self, series, dtype=bool):
        return np.array(self.mlb.transform(series), dtype=dtype)
