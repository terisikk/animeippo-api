import animeippo.recommendation.scoring as scoring
import animeippo.recommendation.engine as engine
import pandas as pd


def test_abstract_scorer_can_be_instantiated():
    class ConcreteScorer(scoring.AbstractScorer):
        def score(self, scoring_target_df, compare_df, encoder):
            return super().score(scoring_target_df, compare_df, encoder)

    filter = ConcreteScorer()
    filter.score(None, None, None)

    assert issubclass(filter.__class__, scoring.AbstractScorer)


def test_genre_similarity_scorer():
    source_df = pd.DataFrame(
        {
            "genres": [["Action", "Adventure"], ["Action", "Fantasy"]],
            "title": ["Bleach", "Fate/Zero"],
        }
    )
    target_df = pd.DataFrame(
        {"genres": [["Romance", "Comedy"], ["Action", "Adventure"]], "title": ["Kaguya", "Naruto"]}
    )

    scorer = scoring.GenreSimilarityScorer()

    recommendations = scorer.score(
        target_df,
        source_df,
        engine.CategoricalEncoder(["Action", "Adventure", "Fantasy"]),
    )

    expected = "Naruto"
    actual = recommendations.iloc[0]["title"]

    assert actual == expected
    assert recommendations.columns.tolist() == ["genres", "title", "recommend_score"]
    assert not recommendations["recommend_score"].isnull().values.any()


def test_genre_similarity_scorer_weighted():
    source_df = pd.DataFrame(
        {
            "genres": [["Action", "Adventure"], ["Fantasy", "Adventure"]],
            "title": ["Bleach", "Fate/Zero"],
            "user_score": [1, 10],
        }
    )
    target_df = pd.DataFrame(
        {
            "genres": [["Action", "Adventure"], ["Fantasy", "Adventure"]],
            "title": ["Naruto", "Inuyasha"],
        }
    )

    scorer = scoring.GenreSimilarityScorer(weighted=True)

    recommendations = scorer.score(
        target_df,
        source_df,
        engine.CategoricalEncoder(["Action", "Adventure", "Fantasy"]),
    )

    expected = "Inuyasha"
    actual = recommendations.iloc[0]["title"]

    assert actual == expected
    assert not recommendations["recommend_score"].isnull().values.any()


def test_cluster_similarity_scorer():
    source_df = pd.DataFrame(
        {
            "genres": [["Action", "Adventure"], ["Action", "Fantasy"]],
            "title": ["Bleach", "Fate/Zero"],
            "cluster": [0, 1],
        }
    )
    target_df = pd.DataFrame(
        {"genres": [["Romance", "Comedy"], ["Action", "Adventure"]], "title": ["Kaguya", "Naruto"]}
    )

    scorer = scoring.ClusterSimilarityScorer(2)

    recommendations = scorer.score(
        target_df,
        source_df,
        engine.CategoricalEncoder(["Action", "Adventure", "Fantasy", "Romance", "Comedy"]),
    )
    expected = "Naruto"
    actual = recommendations.iloc[0]["title"]

    assert actual == expected
    assert recommendations.columns.tolist() == ["genres", "title", "recommend_score"]
    assert not recommendations["recommend_score"].isnull().values.any()


def test_cluster_similarity_scorer_weighted():
    source_df = pd.DataFrame(
        {
            "genres": [["Action", "Adventure"], ["Fantasy", "Adventure"]],
            "title": ["Bleach", "Fate/Zero"],
            "user_score": [10, 1],
            "cluster": [0, 1],
        }
    )
    target_df = pd.DataFrame(
        {
            "genres": [["Fantasy", "Adventure"], ["Action", "Adventure"]],
            "title": ["Inuyasha", "Naruto"],
        }
    )

    scorer = scoring.ClusterSimilarityScorer(2, weighted=True)

    recommendations = scorer.score(
        target_df,
        source_df,
        engine.CategoricalEncoder(["Action", "Adventure", "Fantasy", "Romance", "Comedy"]),
    )
    expected = "Naruto"
    actual = recommendations.iloc[0]["title"]

    assert actual == expected
    assert recommendations.columns.tolist() == ["genres", "title", "recommend_score"]
    assert not recommendations["recommend_score"].isnull().values.any()
