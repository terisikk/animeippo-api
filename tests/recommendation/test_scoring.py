import pandas as pd
import polars as pl

from animeippo.recommendation import scoring, dataset, profile


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
    source_df = pl.DataFrame(
        {
            "id": [1, 2],
            "features": [["Action", "Adventure"], ["Action", "Fantasy"]],
            "title": ["Bleach", "Fate/Zero"],
            "encoded": [[1, 1, 0, 0, 0], [1, 0, 1, 0, 0]],
        },
    )

    target_df = pl.DataFrame(
        {
            "id": [3, 4],
            "features": [["Romance", "Comedy"], ["Action", "Adventure"]],
            "title": ["Kaguya", "Naruto"],
            "encoded": [[0, 0, 0, 1, 1], [1, 1, 0, 0, 0]],
        },
    )

    scorer = scoring.FeaturesSimilarityScorer()

    uprofile = profile.UserProfile("Test", source_df)
    recommendations = target_df.with_columns(
        recommend_score=scorer.score(dataset.RecommendationModel(uprofile, target_df))
    ).sort("recommend_score", descending=True)

    expected = "Naruto"
    actual = recommendations["title"].item(0)

    assert actual == expected
    assert not recommendations["recommend_score"].is_null().any()


def test_feature_similarity_scorer_weighted():
    source_df = pl.DataFrame(
        {
            "id": [1, 2],
            "features": [["Action", "Adventure"], ["Fantasy", "Adventure"]],
            "title": ["Bleach", "Fate/Zero"],
            "encoded": [[1, 1, 0], [0, 1, 1]],
            "score": [1, 10],
        },
    )

    target_df = pl.DataFrame(
        {
            "id": [3, 4],
            "features": [["Action", "Adventure"], ["Fantasy", "Adventure"]],
            "title": ["Naruto", "Inuyasha"],
            "encoded": [[1, 1, 0], [0, 1, 1]],
        },
    )

    scorer = scoring.FeaturesSimilarityScorer(
        weighted=True,
    )

    uprofile = profile.UserProfile("Test", source_df)
    recommendations = target_df.with_columns(
        recommend_score=scorer.score(dataset.RecommendationModel(uprofile, target_df))
    ).sort("recommend_score", descending=True)

    expected = "Inuyasha"
    actual = recommendations["title"].item(0)

    assert actual == expected
    assert not recommendations["recommend_score"].is_null().any()


def test_feature_correlation_scorer():
    source_df = pl.DataFrame(
        {
            "id": [1, 2],
            "features": [["Action", "Adventure"], ["Action", "Fantasy"]],
            "title": ["Bleach", "Fate/Zero"],
            "encoded": [[1, 1, 0, 0, 0], [1, 0, 1, 0, 0]],
            "score": [5, 10],
            "user_status": ["completed", "completed"],
        },
    )

    target_df = pl.DataFrame(
        {
            "id": [3, 4],
            "features": [["Action", "Fantasy"], ["Action", "Adventure"]],
            "title": ["Fate/Grand Order", "Naruto"],
            "encoded": [[1, 0, 1, 0, 0], [1, 1, 0, 0, 0]],
        }
    )

    scorer = scoring.FeatureCorrelationScorer()

    uprofile = profile.UserProfile("Test", source_df)
    recommendations = target_df.with_columns(
        recommend_score=scorer.score(
            dataset.RecommendationModel(
                uprofile,
                target_df,
                features=pl.Series(["Action", "Adventure", "Fantasy", "Romance", "Sci-fi"]),
            )
        )
    ).sort("recommend_score", descending=True)

    expected = "Fate/Grand Order"
    actual = recommendations["title"].item(0)

    assert actual == expected
    assert not recommendations["recommend_score"].is_null().any()


def test_genre_average_scorer():
    # Intermittent failures, order probably not guaranteed somwhere
    # see also studio_correlation_scorer
    source_df = pl.DataFrame(
        {
            "id": [1, 2],
            "genres": [["Action", "Adventure"], ["Fantasy", "Adventure"]],
            "title": ["Bleach", "Fate/Zero"],
            "score": [1, 10],
        }
    )
    target_df = pl.DataFrame(
        {
            "id": [3, 4],
            "genres": [["Action", "Adventure"], ["Fantasy", "Adventure"]],
            "title": ["Naruto", "Inuyasha"],
        }
    )

    scorer = scoring.GenreAverageScorer()

    uprofile = profile.UserProfile("Test", source_df)
    recommendations = target_df.with_columns(
        recommend_score=scorer.score(
            dataset.RecommendationModel(
                uprofile,
                target_df,
            )
        )
    ).sort("recommend_score", descending=True)

    expected = "Inuyasha"
    actual = recommendations["title"].item(0)

    assert actual == expected
    assert not recommendations["recommend_score"].is_null().any()


def test_cluster_similarity_scorer():
    source_df = pl.DataFrame(
        {
            "id": [1, 2],
            "features": [["Action", "Adventure"], ["Action", "Fantasy"]],
            "title": ["Bleach", "Fate/Zero"],
            "score": [10, 10],
            "cluster": [1, 0],
        },
    )

    target_df = pl.DataFrame(
        {
            "id": [3, 4],
            "features": [["Romance", "Comedy"], ["Action", "Adventure"]],
            "title": ["Kaguya", "Naruto"],
        }
    )

    scorer = scoring.ClusterSimilarityScorer()

    uprofile = profile.UserProfile("Test", source_df)
    data = dataset.RecommendationModel(
        uprofile,
        target_df,
    )

    data.similarity_matrix = pl.DataFrame(
        {
            "3": [0, 0],
            "4": [1, 0.5],
            "id": [1, 2],
        }
    )

    recommendations = target_df.with_columns(recommend_score=scorer.score(data)).sort(
        "recommend_score", descending=True
    )

    expected = "Naruto"
    actual = recommendations["title"].item(0)

    assert actual == expected
    assert not recommendations["recommend_score"].is_null().any()


def test_cluster_similarity_scorer_weighted():
    source_df = pl.DataFrame(
        {
            "id": [1, 2],
            "features": [["Action", "Adventure"], ["Fantasy", "Adventure"]],
            "title": ["Bleach", "Fate/Zero"],
            "score": [10, 1],
            "cluster": [0, 1],
        }
    )
    target_df = pl.DataFrame(
        {
            "id": [3, 4],
            "features": [["Fantasy", "Adventure"], ["Action", "Adventure"]],
            "title": ["Inuyasha", "Naruto"],
        }
    )

    scorer = scoring.ClusterSimilarityScorer(weighted=True)

    uprofile = profile.UserProfile("Test", source_df)
    data = dataset.RecommendationModel(
        uprofile,
        target_df,
    )
    data.similarity_matrix = pl.DataFrame(
        {
            "3": [0.5, 1],
            "4": [1, 0.5],
            "id": [1, 2],
        }
    )

    recommendations = target_df.with_columns(recommend_score=scorer.score(data)).sort(
        "recommend_score", descending=True
    )

    expected = "Naruto"
    actual = recommendations["title"].item(0)

    assert actual == expected
    assert not recommendations["recommend_score"].is_null().any()


def test_studio_count_scorer():
    source_df = pl.DataFrame(
        {
            "studios": [["MAPPA"], ["Kinema Citrus", "GIFTanimation", "Studio Jemi"]],
            "title": ["Vinland Saga", "Cardfight!! Vanguard"],
        }
    )
    target_df = pl.DataFrame(
        {
            "studios": [["Bones"], ["MAPPA"]],
            "title": ["Bungou Stray Dogs", "Jujutsu Kaisen"],
        }
    )

    scorer = scoring.StudioCountScorer()

    uprofile = profile.UserProfile("Test", source_df)
    recommendations = target_df.with_columns(
        recommend_score=scorer.score(
            dataset.RecommendationModel(
                uprofile,
                target_df,
            )
        )
    ).sort("recommend_score", descending=True)

    expected = "Jujutsu Kaisen"
    actual = recommendations["title"].item(0)

    assert actual == expected
    assert not recommendations["recommend_score"].is_null().any()


def test_studio_count_scorer_does_not_fail_with_zero_studios():
    source_df = pl.DataFrame(
        {
            "studios": [["MAPPA"], ["Kinema Citrus", "GIFTanimation", "Studio Jemi"]],
            "title": ["Vinland Saga", "Cardfight!! Vanguard"],
        }
    )
    target_df = pl.DataFrame(
        {
            "studios": [[], ["MAPPA"]],
            "title": ["Bungou Stray Dogs", "Jujutsu Kaisen"],
        }
    )

    scorer = scoring.StudioCountScorer()

    uprofile = profile.UserProfile("Test", source_df)
    recommendations = target_df.with_columns(
        recommend_score=scorer.score(
            dataset.RecommendationModel(
                uprofile,
                target_df,
            )
        )
    ).sort("recommend_score", descending=True)

    assert not recommendations["recommend_score"].is_null().any()


def test_studio_correlation_scorer():
    source_df = pl.DataFrame(
        {
            "id": [1, 2],
            "studios": [["MAPPA"], ["Bones"]],
            "title": ["Vinland Saga", "Fullmetal Alchemist: Brotherhood"],
            "score": [10, 1],
        }
    )
    target_df = pl.DataFrame(
        {
            "id": [3, 4],
            "studios": [["Bones"], ["MAPPA"]],
            "title": ["Bungou Stray Dogs", "Jujutsu Kaisen"],
        }
    )

    scorer = scoring.StudioCorrelationScorer()

    uprofile = profile.UserProfile("Test", source_df)
    recommendations = target_df.with_columns(
        recommend_score=scorer.score(
            dataset.RecommendationModel(
                uprofile,
                target_df,
            )
        )
    ).sort("recommend_score", descending=True)

    expected = "Jujutsu Kaisen"
    actual = recommendations["title"].item(0)

    assert actual == expected
    assert not recommendations["recommend_score"].is_null().any()


def test_popularity_scorer():
    target_df = pl.DataFrame(
        {
            "studios": [["Bones"], ["MAPPA"]],
            "title": ["Bungou Stray Dogs", "Jujutsu Kaisen"],
            "popularity": [10, 100],
        }
    )

    scorer = scoring.PopularityScorer()

    recommendations = target_df.with_columns(
        recommend_score=scorer.score(
            dataset.RecommendationModel(
                None,
                target_df,
            )
        )
    ).sort("recommend_score", descending=True)

    expected = "Jujutsu Kaisen"
    actual = recommendations["title"].item(0)

    assert actual == expected
    assert not recommendations["recommend_score"].is_null().any()


def test_continuation_scorer():
    compare = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "title": ["Anime A", "Anime B", "Anime B Spinoff", "Anime C"],
            "user_status": ["completed", "completed", "completed", "completed"],
            "score": [8, 6, 7, 9],
        }
    )

    original = pl.DataFrame(
        {
            "id": [5, 6, 7, 8],
            "title": ["Anime A Season 2", "Anime E Season 2", "Anime B Season 2", "Anime F"],
            "continuation_to": [[1], [9], [2, 3], []],
        }
    )

    scorer = scoring.ContinuationScorer()

    uprofile = profile.UserProfile("Test", compare)
    actual = scorer.score(dataset.RecommendationModel(uprofile, original))

    assert actual.to_list() == [0.8, 0.0, 0.7, 0.0]


def test_continuation_scorer_scores_nan_with_zero():
    compare = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "title": ["Anime A", "Anime B", "Anime B Spinoff", "Anime C"],
            "user_status": ["completed", "completed", "completed", "completed"],
            "score": [None, 6, 7, 8],
        }
    )

    original = pl.DataFrame(
        {
            "id": [5, 6, 7, 8],
            "title": ["Anime A Season 2", "Anime E Season 2", "Anime B Season 2", "Anime F"],
            "continuation_to": [[1], [9], [2, 3], []],
        }
    )

    scorer = scoring.ContinuationScorer()

    uprofile = profile.UserProfile("Test", compare)
    actual = scorer.score(dataset.RecommendationModel(uprofile, original))

    assert actual.to_list() == [0.7, 0.0, 0.7, 0.0]


def test_continuation_scorer_takes_max_of_duplicate_relations():
    compare = pl.DataFrame(
        {
            "id": [1, 2],
            "title": ["Anime A", "Anime A Spinoff"],
            "user_status": ["completed", "completed"],
            "score": [2, 8],
        }
    )

    original = pl.DataFrame(
        {
            "id": [5],
            "title": ["Anime A Season 2"],
            "continuation_to": [[1, 2]],
        }
    )

    scorer = scoring.ContinuationScorer()

    uprofile = profile.UserProfile("Test", compare)
    actual = scorer.score(dataset.RecommendationModel(uprofile, original))

    assert actual.to_list() == [0.8]


def test_source_scorer():
    compare = pl.DataFrame(
        {
            "title": ["Anime A", "Anime B"],
            "source": ["original", "manga"],
            "score": [5, 10],
        }
    )

    target_df = pl.DataFrame(
        {
            "title": ["Bungou Stray Dogs", "Jujutsu Kaisen"],
            "source": ["original", "manga"],
        }
    )

    scorer = scoring.SourceScorer()

    uprofile = profile.UserProfile("Test", compare)
    recommendations = target_df.with_columns(
        recommend_score=scorer.score(
            dataset.RecommendationModel(
                uprofile,
                target_df,
            )
        )
    ).sort("recommend_score", descending=True)

    expected = "Jujutsu Kaisen"
    actual = recommendations["title"].item(0)

    assert actual == expected
    assert not recommendations["recommend_score"].is_null().any()


def test_direct_similarity_scorer():
    source_df = pl.DataFrame(
        {
            "id": [1, 2],
            "features": [["Action", "Adventure"], ["Action", "Fantasy"]],
            "title": ["Bleach", "Fate/Zero"],
            "score": [10, 9],
        },
    )
    target_df = pl.DataFrame(
        {
            "id": [3, 4],
            "features": [["Fantasy", "Romance", "Comedy"], ["Action", "Adventure"]],
            "title": ["Kaguya", "Naruto"],
        },
    )

    scorer = scoring.DirectSimilarityScorer()

    uprofile = profile.UserProfile("Test", source_df)
    data = dataset.RecommendationModel(
        uprofile,
        target_df,
    )

    data.similarity_matrix = pl.DataFrame(
        {
            "3": [0, 0],
            "4": [1, 0.5],
            "id": [1, 2],
        }
    )

    recommendations = target_df.with_columns(recommend_score=scorer.score(data)).sort(
        "recommend_score", descending=True
    )

    expected = "Naruto"
    actual = recommendations["title"].item(0)

    assert actual == expected
    assert not recommendations["recommend_score"].is_null().any()


def test_adaptation_scorer():
    compare = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "title": ["Manga A", "Manga B", "Manga C"],
            "score": [None, 8, 4],
        }
    )

    target = pl.DataFrame(
        {
            "id": [5, 6, 7],
            "title": ["Anime X", "Anime A", "Anime Y"],
            "adaptation_of": [[10], [1], []],
        }
    )

    scorer = scoring.AdaptationScorer()

    data = dataset.RecommendationModel(None, target)

    actual = scorer.score(data)

    assert actual is None

    data.mangalist = compare

    actual = target.with_columns(recommend_score=scorer.score(data)).sort(
        "recommend_score", descending=True
    )

    assert actual["title"].to_list() == ["Anime A", "Anime X", "Anime Y"]
    assert actual.filter(pl.col("id") == 7)["recommend_score"].item() == 0


def test_format_scorer():
    target = pl.DataFrame(
        {
            "title": ["Anime A", "Anime B", "Anime C"],
            "format": ["TV_SHORT", "TV", "TV"],
            "episodes": [5, 12, 12],
            "duration": [5, 25, 25],
        }
    )

    scorer = scoring.FormatScorer()

    recommendations = target.with_columns(
        recommend_score=scorer.score(
            dataset.RecommendationModel(
                None,
                target,
            )
        )
    ).sort("recommend_score", descending=True)

    assert recommendations["title"].to_list() == ["Anime B", "Anime C", "Anime A"]


def test_director_correlation_scorer():
    source_df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "directors": [["1"], ["2"], ["3"]],
            "title": ["Vinland Saga", "Fullmetal Alchemist: Brotherhood", "Bleach"],
            "score": [10, 1, 2],
        }
    )
    target_df = pl.DataFrame(
        {
            "id": [4, 5],
            "directors": [["4"], ["1"]],
            "title": ["Bungou Stray Dogs", "Jujutsu Kaisen"],
        }
    )

    scorer = scoring.DirectorCorrelationScorer()

    uprofile = profile.UserProfile("Test", source_df)
    recommendations = target_df.with_columns(
        recommend_score=scorer.score(
            dataset.RecommendationModel(
                uprofile,
                target_df,
            )
        )
    ).sort("recommend_score", descending=True)

    expected = "Jujutsu Kaisen"
    actual = recommendations["title"].item(0)

    assert actual == expected
    assert not recommendations["recommend_score"].is_null().any()
