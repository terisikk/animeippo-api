import animeippo.recommendation.scoring as scoring
import pandas as pd


def test_abstract_scorer_can_be_instantiated():
    class ConcreteScorer(scoring.AbstractScorer):
        def name(self):
            super().name

        def score(self, scoring_target_df, compare_df):
            return super().score(scoring_target_df, compare_df)

    filter = ConcreteScorer()
    filter.score(None, None)
    filter.name()

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

    scorer = scoring.GenreSimilarityScorer(
        scoring.CategoricalEncoder(["Action", "Adventure", "Fantasy", "Romance", "Comedy"])
    )

    target_df["recommend_score"] = scorer.score(
        target_df,
        source_df,
    )

    recommendations = target_df.sort_values("recommend_score", ascending=False)

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
            "score": [1, 10],
        }
    )
    target_df = pd.DataFrame(
        {
            "genres": [["Action", "Adventure"], ["Fantasy", "Adventure"]],
            "title": ["Naruto", "Inuyasha"],
        }
    )

    scorer = scoring.GenreSimilarityScorer(
        scoring.CategoricalEncoder(["Action", "Adventure", "Fantasy"]), weighted=True
    )

    target_df["recommend_score"] = scorer.score(
        target_df,
        source_df,
    )

    recommendations = target_df.sort_values("recommend_score", ascending=False)

    expected = "Inuyasha"
    actual = recommendations.iloc[0]["title"]

    assert actual == expected
    assert not recommendations["recommend_score"].isnull().values.any()


def test_genre_average_scorer():
    source_df = pd.DataFrame(
        {
            "genres": [["Action", "Adventure"], ["Fantasy", "Adventure"]],
            "title": ["Bleach", "Fate/Zero"],
            "score": [1, 10],
        }
    )
    target_df = pd.DataFrame(
        {
            "genres": [["Action", "Adventure"], ["Fantasy", "Adventure"]],
            "title": ["Naruto", "Inuyasha"],
        }
    )

    scorer = scoring.GenreAverageScorer()

    target_df["recommend_score"] = scorer.score(
        target_df,
        source_df,
    )

    recommendations = target_df.sort_values("recommend_score", ascending=False)

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

    scorer = scoring.ClusterSimilarityScorer(
        scoring.CategoricalEncoder(["Action", "Adventure", "Fantasy", "Romance", "Comedy"]), 2
    )

    target_df["recommend_score"] = scorer.score(
        target_df,
        source_df,
    )

    recommendations = target_df.sort_values("recommend_score", ascending=False)

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
            "score": [10, 1],
            "cluster": [0, 1],
        }
    )
    target_df = pd.DataFrame(
        {
            "genres": [["Fantasy", "Adventure"], ["Action", "Adventure"]],
            "title": ["Inuyasha", "Naruto"],
        }
    )

    scorer = scoring.ClusterSimilarityScorer(
        scoring.CategoricalEncoder(["Action", "Adventure", "Fantasy", "Romance", "Comedy"]),
        2,
        weighted=True,
    )

    target_df["recommend_score"] = scorer.score(
        target_df,
        source_df,
    )

    recommendations = target_df.sort_values("recommend_score", ascending=False)

    expected = "Naruto"
    actual = recommendations.iloc[0]["title"]

    assert actual == expected
    assert recommendations.columns.tolist() == ["genres", "title", "recommend_score"]
    assert not recommendations["recommend_score"].isnull().values.any()


def test_studio_count_scorer():
    source_df = pd.DataFrame(
        {
            "studios": [["MAPPA"], ["Kinema Citrus", "GIFTanimation", "Studio Jemi"]],
            "title": ["Vinland Saga", "Cardfight!! Vanguard"],
        }
    )
    target_df = pd.DataFrame(
        {
            "studios": [["Bones"], ["MAPPA"]],
            "title": ["Bungou Stray Dogs", "Jujutsu Kaisen"],
        }
    )

    scorer = scoring.StudioCountScorer()

    target_df["recommend_score"] = scorer.score(
        target_df,
        source_df,
    )

    recommendations = target_df.sort_values("recommend_score", ascending=False)

    expected = "Jujutsu Kaisen"
    actual = recommendations.iloc[0]["title"]

    assert actual == expected
    assert "recommend_score" in recommendations.columns.tolist()
    assert not recommendations["recommend_score"].isnull().values.any()


def test_studio_average_scorer():
    source_df = pd.DataFrame(
        {
            "studios": [["MAPPA"], ["Bones"]],
            "title": ["Vinland Saga", "Fullmetal Alchemist: Brotherhood"],
            "score": [10, 1],
        }
    )
    target_df = pd.DataFrame(
        {
            "studios": [["Bones"], ["MAPPA"]],
            "title": ["Bungou Stray Dogs", "Jujutsu Kaisen"],
        }
    )

    scorer = scoring.StudioAverageScorer()

    target_df["recommend_score"] = scorer.score(
        target_df,
        source_df,
    )

    recommendations = target_df.sort_values("recommend_score", ascending=False)

    expected = "Jujutsu Kaisen"
    actual = recommendations.iloc[0]["title"]

    assert actual == expected
    assert "recommend_score" in recommendations.columns.tolist()
    assert not recommendations["recommend_score"].isnull().values.any()


def test_popularity_scorer():
    target_df = pd.DataFrame(
        {
            "studios": [["Bones"], ["MAPPA"]],
            "title": ["Bungou Stray Dogs", "Jujutsu Kaisen"],
            "num_list_users": [10, 100],
        }
    )

    scorer = scoring.PopularityScorer()

    target_df["recommend_score"] = scorer.score(
        target_df,
        None,
    )

    recommendations = target_df.sort_values("recommend_score", ascending=False)

    expected = "Jujutsu Kaisen"
    actual = recommendations.iloc[0]["title"]

    assert actual == expected
    assert "recommend_score" in recommendations.columns.tolist()
    assert not recommendations["recommend_score"].isnull().values.any()
