import pandas as pd

from animeippo.recommendation import scoring, dataset


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


def test_feature_similarity_scorer():
    source_df = pd.DataFrame(
        {
            "features": [["Action", "Adventure"], ["Action", "Fantasy"]],
            "title": ["Bleach", "Fate/Zero"],
            "encoded": [[True, True, False, False, False], [True, False, True, False, False]],
        },
    )

    target_df = pd.DataFrame(
        {
            "features": [["Romance", "Comedy"], ["Action", "Adventure"]],
            "title": ["Kaguya", "Naruto"],
            "encoded": [[False, False, False, True, True], [True, True, False, False, False]],
        }
    )

    scorer = scoring.FeaturesSimilarityScorer()

    target_df["recommend_score"] = scorer.score(dataset.UserDataSet(source_df, target_df))

    recommendations = target_df.sort_values("recommend_score", ascending=False)

    expected = "Naruto"
    actual = recommendations.iloc[0]["title"]

    assert actual == expected
    assert not recommendations["recommend_score"].isnull().values.any()


def test_feature_correlation_scorer():
    source_df = pd.DataFrame(
        {
            "features": [["Action", "Adventure"], ["Action", "Fantasy"]],
            "title": ["Bleach", "Fate/Zero"],
            "encoded": [[True, True, False, False, False], [True, False, True, False, False]],
            "score": [5, 10],
        },
    )

    target_df = pd.DataFrame(
        {
            "features": [["Action", "Fantasy"], ["Action", "Adventure"]],
            "title": ["Fate/Grand Order", "Naruto"],
            "encoded": [[True, False, True, False, False], [True, True, False, False, False]],
        }
    )

    scorer = scoring.FeatureCorrelationScorer()

    target_df["recommend_score"] = scorer.score(
        dataset.UserDataSet(
            source_df,
            target_df,
            features=pd.Series(["Action", "Adventure", "Fantasy", "Romance", "Sci-fi"]),
        )
    )

    recommendations = target_df.sort_values("recommend_score", ascending=False)

    expected = "Fate/Grand Order"
    actual = recommendations.iloc[0]["title"]

    assert actual == expected
    assert not recommendations["recommend_score"].isnull().values.any()


def test_feature_similarity_scorer_weighted():
    source_df = pd.DataFrame(
        {
            "features": [["Action", "Adventure"], ["Fantasy", "Adventure"]],
            "title": ["Bleach", "Fate/Zero"],
            "encoded": [[True, True, False], [False, True, True]],
            "score": [1, 10],
        }
    )
    target_df = pd.DataFrame(
        {
            "features": [["Action", "Adventure"], ["Fantasy", "Adventure"]],
            "title": ["Naruto", "Inuyasha"],
            "encoded": [[True, True, False], [False, True, True]],
        }
    )

    scorer = scoring.FeaturesSimilarityScorer(
        weighted=True,
    )

    target_df["recommend_score"] = scorer.score(dataset.UserDataSet(source_df, target_df))

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

    target_df["recommend_score"] = scorer.score(dataset.UserDataSet(source_df, target_df))

    recommendations = target_df.sort_values("recommend_score", ascending=False)

    expected = "Inuyasha"
    actual = recommendations.iloc[0]["title"]

    assert actual == expected
    assert not recommendations["recommend_score"].isnull().values.any()


def test_cluster_similarity_scorer():
    encoded1 = [
        [True, True, False, False, False],
        [True, False, True, False, False],
    ]
    encoded2 = [
        [False, False, False, True, True],
        [True, True, False, False, False],
    ]

    source_df = pd.DataFrame(
        {
            "features": [["Action", "Adventure"], ["Action", "Fantasy"]],
            "title": ["Bleach", "Fate/Zero"],
            "encoded": encoded1,
            "cluster": [1, 0],
        },
    )

    target_df = pd.DataFrame(
        {
            "features": [["Romance", "Comedy"], ["Action", "Adventure"]],
            "title": ["Kaguya", "Naruto"],
            "encoded": encoded2,
        }
    )

    scorer = scoring.ClusterSimilarityScorer()

    target_df["recommend_score"] = scorer.score(dataset.UserDataSet(source_df, target_df))

    recommendations = target_df.sort_values("recommend_score", ascending=False)

    expected = "Naruto"
    actual = recommendations.iloc[0]["title"]

    assert actual == expected
    assert not recommendations["recommend_score"].isnull().values.any()


def test_cluster_similarity_scorer_weighted():
    source_df = pd.DataFrame(
        {
            "features": [["Action", "Adventure"], ["Fantasy", "Adventure"]],
            "title": ["Bleach", "Fate/Zero"],
            "score": [10, 1],
            "cluster": [0, 1],
            "encoded": [[True, True, False], [False, True, True]],
        }
    )
    target_df = pd.DataFrame(
        {
            "features": [["Fantasy", "Adventure"], ["Action", "Adventure"]],
            "title": ["Inuyasha", "Naruto"],
            "encoded": [[False, True, True], [True, True, False]],
        }
    )

    scorer = scoring.ClusterSimilarityScorer(weighted=True)

    target_df["recommend_score"] = scorer.score(dataset.UserDataSet(source_df, target_df))

    recommendations = target_df.sort_values("recommend_score", ascending=False)

    expected = "Naruto"
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

    target_df["recommend_score"] = scorer.score(dataset.UserDataSet(source_df, target_df))

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

    target_df["recommend_score"] = scorer.score(dataset.UserDataSet(source_df, target_df))

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

    scorer = scoring.StudioCorrelationScorer()

    target_df["recommend_score"] = scorer.score(dataset.UserDataSet(source_df, target_df))

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

    target_df["recommend_score"] = scorer.score(dataset.UserDataSet(None, target_df))

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
            "user_status": ["completed", "completed", "completed", "completed"],
            "score": [8, 6, 7, 9],
        }
    )
    compare = compare.set_index("id")

    original = pd.DataFrame(
        {
            "id": [5, 6, 7, 8],
            "title": ["Anime A Season 2", "Anime E Season 2", "Anime B Season 2", "Anime F"],
            "continuation_to": [[1], [9], [2, 3], []],
        }
    )
    original = original.set_index("id")

    scorer = scoring.ContinuationScorer()

    actual = scorer.score(dataset.UserDataSet(compare, original))

    assert actual.to_list() == [0.8, 0, 0.7, 0]


def test_continuation_scorer_scores_nan_with_zero():
    compare = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "title": ["Anime A", "Anime B", "Anime B Spinoff", "Anime C"],
            "user_status": ["completed", "completed", "completed", "completed"],
            "score": [pd.NA, 6, 7, 8],
        }
    )
    compare = compare.set_index("id")

    original = pd.DataFrame(
        {
            "id": [5, 6, 7, 8],
            "title": ["Anime A Season 2", "Anime E Season 2", "Anime B Season 2", "Anime F"],
            "continuation_to": [[1], [9], [2, 3], []],
        }
    )
    original = original.set_index("id")

    scorer = scoring.ContinuationScorer()

    actual = scorer.score(dataset.UserDataSet(compare, original))

    assert actual.to_list() == [0.7, 0, 0.7, 0]


def test_continuation_scorer_takes_max_of_duplicate_relations():
    compare = pd.DataFrame(
        {
            "id": [1, 2],
            "title": ["Anime A", "Anime A Spinoff"],
            "user_status": ["completed", "completed"],
            "score": [2, 8],
        }
    )
    compare = compare.set_index("id")

    original = pd.DataFrame(
        {
            "id": [5],
            "title": ["Anime A Season 2"],
            "continuation_to": [[1, 2]],
        }
    )
    original = original.set_index("id")

    scorer = scoring.ContinuationScorer()

    actual = scorer.score(dataset.UserDataSet(compare, original))

    assert actual.to_list() == [0.8]


def test_source_scorer():
    compare = pd.DataFrame(
        {
            "title": ["Anime A", "Anime B"],
            "source": ["original", "manga"],
            "score": [5, 10],
        }
    )

    actual = pd.DataFrame(
        {
            "title": ["Bungou Stray Dogs", "Jujutsu Kaisen"],
            "source": ["original", "manga"],
        }
    )

    scorer = scoring.SourceScorer()

    actual["recommend_score"] = scorer.score(dataset.UserDataSet(compare, actual))

    recommendations = actual.sort_values("recommend_score", ascending=False)

    expected = "Jujutsu Kaisen"
    actual = recommendations.iloc[0]["title"]

    assert actual == expected
    assert not recommendations["recommend_score"].isnull().values.any()


def test_direct_similarity_scorer():
    source_df = pd.DataFrame(
        {
            "features": [["Action", "Adventure"], ["Action", "Fantasy"]],
            "title": ["Bleach", "Fate/Zero"],
            "score": [10, 10],
            "encoded": [[True, True, False, False, False], [True, False, True, False, False]],
        }
    )
    target_df = pd.DataFrame(
        {
            "features": [["Romance", "Comedy"], ["Action", "Adventure"]],
            "title": ["Kaguya", "Naruto"],
            "encoded": [[False, False, False, True, True], [True, True, False, False, False]],
        }
    )

    scorer = scoring.DirectSimilarityScorer()

    target_df["recommend_score"] = scorer.score(dataset.UserDataSet(source_df, target_df))

    recommendations = target_df.sort_values("recommend_score", ascending=False)

    expected = "Naruto"
    actual = recommendations.iloc[0]["title"]

    assert actual == expected
    assert not recommendations["recommend_score"].isnull().values.any()


def test_adaptation_scorer():
    compare = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "title": ["Manga A", "Manga B", "Manga C"],
            "score": [pd.NA, 8, 4],
        }
    )
    compare = compare.set_index("id")

    target = pd.DataFrame(
        {
            "id": [5, 6, 7],
            "title": ["Anime X", "Anime A", "Anime Y"],
            "adaptation_of": [[10], [1], []],
        }
    )
    target = target.set_index("id")

    scorer = scoring.AdaptationScorer()

    data = dataset.UserDataSet(None, target)

    actual = scorer.score(data)

    assert pd.isna(actual)

    data.mangalist = compare

    target["recommend_score"] = scorer.score(data)

    actual = target.sort_values("recommend_score", ascending=False)

    assert actual["title"].tolist() == ["Anime A", "Anime X", "Anime Y"]
    assert actual.loc[7, "recommend_score"] == 0


def test_format_scorer():
    target = pd.DataFrame(
        {
            "title": ["Anime A", "Anime B", "Anime C"],
            "format": ["TV_SHORT", "TV", "TV"],
            "episodes": [5, 12, 12],
            "duration": [5, 25, 25],
        }
    )

    scorer = scoring.FormatScorer()

    target["recommend_score"] = scorer.score(dataset.UserDataSet(None, target))

    actual = target.sort_values("recommend_score", ascending=False)

    assert actual["title"].tolist() == ["Anime B", "Anime C", "Anime A"]


def test_director_correlation_scorer():
    source_df = pd.DataFrame(
        {
            "directors": [["1"], ["2"]],
            "title": ["Vinland Saga", "Fullmetal Alchemist: Brotherhood"],
            "score": [10, 1],
        }
    )
    target_df = pd.DataFrame(
        {
            "directors": [["3"], ["1"]],
            "title": ["Bungou Stray Dogs", "Jujutsu Kaisen"],
        }
    )

    scorer = scoring.DirectorCorrelationScorer()

    target_df["recommend_score"] = scorer.score(dataset.UserDataSet(source_df, target_df))

    recommendations = target_df.sort_values("recommend_score", ascending=False)

    expected = "Jujutsu Kaisen"
    actual = recommendations.iloc[0]["title"]

    assert actual == expected
    assert not recommendations["recommend_score"].isnull().values.any()
