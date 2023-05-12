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

    scorer = scoring.FeaturesSimilarityScorer(["genres"])

    target_df["recommend_score"] = scorer.score(
        target_df,
        source_df,
    )

    recommendations = target_df.sort_values("recommend_score", ascending=False)

    expected = "Naruto"
    actual = recommendations.iloc[0]["title"]

    assert actual == expected
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

    scorer = scoring.FeaturesSimilarityScorer(
        ["genres"],
        weighted=True,
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
        }
    )
    target_df = pd.DataFrame(
        {"genres": [["Romance", "Comedy"], ["Action", "Adventure"]], "title": ["Kaguya", "Naruto"]}
    )

    scorer = scoring.ClusterSimilarityScorer(["genres"])

    target_df["recommend_score"] = scorer.score(
        target_df,
        source_df,
    )

    recommendations = target_df.sort_values("recommend_score", ascending=False)

    expected = "Naruto"
    actual = recommendations.iloc[0]["title"]

    assert actual == expected
    assert not recommendations["recommend_score"].isnull().values.any()


def test_cluster_similarity_scorer_weighted():
    source_df = pd.DataFrame(
        {
            "genres": [["Action", "Adventure"], ["Fantasy", "Adventure"]],
            "title": ["Bleach", "Fate/Zero"],
            "score": [10, 1],
        }
    )
    target_df = pd.DataFrame(
        {
            "genres": [["Fantasy", "Adventure"], ["Action", "Adventure"]],
            "title": ["Inuyasha", "Naruto"],
        }
    )

    scorer = scoring.ClusterSimilarityScorer(["genres"], weighted=True)

    target_df["recommend_score"] = scorer.score(
        target_df,
        source_df,
    )

    recommendations = target_df.sort_values("recommend_score", ascending=False)

    expected = "Inuyasha"
    actual = recommendations.iloc[0]["title"]

    assert actual == expected
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
    assert not recommendations["recommend_score"].isnull().values.any()


def test_studio_count_scorer_does_not_fail_with_zero_studios():
    source_df = pd.DataFrame(
        {
            "studios": [["MAPPA"], ["Kinema Citrus", "GIFTanimation", "Studio Jemi"]],
            "title": ["Vinland Saga", "Cardfight!! Vanguard"],
        }
    )
    target_df = pd.DataFrame(
        {
            "studios": [[], ["MAPPA"]],
            "title": ["Bungou Stray Dogs", "Jujutsu Kaisen"],
        }
    )

    scorer = scoring.StudioCountScorer()

    target_df["recommend_score"] = scorer.score(
        target_df,
        source_df,
    )

    recommendations = target_df.sort_values("recommend_score", ascending=False)

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
    assert not recommendations["recommend_score"].isnull().values.any()


def test_popularity_scorer():
    target_df = pd.DataFrame(
        {
            "studios": [["Bones"], ["MAPPA"]],
            "title": ["Bungou Stray Dogs", "Jujutsu Kaisen"],
            "popularity": [10, 100],
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
    assert not recommendations["recommend_score"].isnull().values.any()


def test_continuation_scorer():
    compare = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "title": ["Anime A", "Anime B", "Anime B Spinoff", "Anime C"],
            "status": ["completed", "completed", "completed", "completed"],
            "score": [8, 6, 7, 9],
        }
    )
    compare = compare.set_index("id")

    original = pd.DataFrame(
        {
            "id": [5, 6, 7, 8],
            "title": ["Anime A Season 2", "Anime E Season 2", "Anime B Season 2", "Anime F"],
            "related_anime": [[1], [9], [2, 3], []],
        }
    )
    original = original.set_index("id")

    scorer = scoring.ContinuationScorer()

    actual = scorer.score(original, compare)

    assert actual.to_list() == [0.8, 0.6, 0.7, 0.6]


def test_continuation_scorer_scores_nan_with_mean_and_new_with_six():
    compare = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "title": ["Anime A", "Anime B", "Anime B Spinoff", "Anime C"],
            "status": ["completed", "completed", "completed", "completed"],
            "score": [pd.NA, 6, 7, 8],
        }
    )
    compare = compare.set_index("id")

    original = pd.DataFrame(
        {
            "id": [5, 6, 7, 8],
            "title": ["Anime A Season 2", "Anime E Season 2", "Anime B Season 2", "Anime F"],
            "related_anime": [[1], [9], [2, 3], []],
        }
    )
    original = original.set_index("id")

    scorer = scoring.ContinuationScorer()

    actual = scorer.score(original, compare)

    assert actual.to_list() == [0.7, 0.6, 0.7, 0.6]


def test_source_scorer():
    compare = pd.DataFrame(
        {"title": ["Anime A", "Anime B"], "source": ["original", "manga"], "score": [5, 10]}
    )

    actual = pd.DataFrame(
        {
            "title": ["Bungou Stray Dogs", "Jujutsu Kaisen"],
            "source": ["original", "manga"],
        }
    )

    scorer = scoring.SourceScorer()

    actual["recommend_score"] = scorer.score(actual, compare)

    recommendations = actual.sort_values("recommend_score", ascending=False)

    expected = "Jujutsu Kaisen"
    actual = recommendations.iloc[0]["title"]

    assert actual == expected
    assert not recommendations["recommend_score"].isnull().values.any()
