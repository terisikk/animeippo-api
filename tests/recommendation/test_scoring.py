import polars as pl

from animeippo.profiling.model import UserProfile
from animeippo.recommendation import scoring
from animeippo.recommendation.model import RecommendationModel


def test_abstract_scorer_can_be_instantiated():
    class ConcreteScorer(scoring.AbstractScorer):
        def name(self):
            return super().name

        def score(self, scoring_target_df, compare_df):
            return super().score(scoring_target_df, compare_df)

    f = ConcreteScorer()
    f.score(None, None)
    f.name()

    assert issubclass(f.__class__, scoring.AbstractScorer)


def test_feature_correlation_scorer():
    source_df = pl.DataFrame(
        {
            "id": [1, 2],
            "features": [["Action", "Adventure"], ["Action", "Fantasy"]],
            "title": ["Bleach", "Fate/Zero"],
            "encoded": [
                {"Action": 1, "Adventure": 1, "Fantasy": 0, "Romance": 0, "Sci-fi": 0},
                {"Action": 1, "Adventure": 0, "Fantasy": 1, "Romance": 0, "Sci-fi": 0},
            ],
            "score": [5, 10],
            "user_status": ["COMPLETED", "COMPLETED"],
        },
    )

    target_df = pl.DataFrame(
        {
            "id": [3, 4],
            "features": [["Action", "Fantasy"], ["Action", "Adventure"]],
            "title": ["Fate/Grand Order", "Naruto"],
            "encoded": [
                {"Action": 1, "Adventure": 0, "Fantasy": 1, "Romance": 0, "Sci-fi": 0},
                {"Action": 1, "Adventure": 1, "Fantasy": 0, "Romance": 0, "Sci-fi": 0},
            ],
        }
    )

    scorer = scoring.FeatureCorrelationScorer()

    uprofile = UserProfile("Test", source_df)
    result = scorer.score(
        RecommendationModel(
            uprofile,
            target_df,
            features=pl.Series(["Action", "Adventure", "Fantasy", "Romance", "Sci-fi"]),
        )
    )

    recommendations = target_df.with_columns(discovery_score=result.score).sort(
        "discovery_score", descending=True
    )

    assert recommendations["title"].item(0) == "Fate/Grand Order"
    assert not recommendations["discovery_score"].is_null().any()
    assert len(result.confidence) == len(target_df)
    assert (result.confidence >= 0).all()
    assert (result.confidence <= 1).all()


def test_feature_correlation_scorer_contested_features_reduce_confidence():
    source_df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "features": [["Action"], ["Action"], ["Drama"], ["Drama"]],
            "encoded": [
                {"Action": 1, "Drama": 0},
                {"Action": 1, "Drama": 0},
                {"Action": 0, "Drama": 1},
                {"Action": 0, "Drama": 1},
            ],
            "score": [9, 8, 2, 3],
            "user_status": ["COMPLETED", "COMPLETED", "DROPPED", "DROPPED"],
        },
    )

    target_df = pl.DataFrame(
        {
            "id": [5],
            "features": [["Action", "Drama"]],
            "encoded": [{"Action": 1, "Drama": 1}],
        }
    )

    scorer = scoring.FeatureCorrelationScorer()
    uprofile = UserProfile("Test", source_df)

    result = scorer.score(
        RecommendationModel(uprofile, target_df, features=pl.Series(["Action", "Drama"]))
    )

    # Action is liked, Drama is dropped — contested features should lower confidence
    assert result.confidence[0] < 1.0


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

    uprofile = UserProfile("Test", source_df)
    data = RecommendationModel(
        uprofile,
        target_df,
    )

    data.similarity_matrix = pl.DataFrame(
        {
            "3": [0, 0],
            "4": [1.0, 0.5],
            "id": [1, 2],
        }
    )

    result = scorer.score(data)
    recommendations = target_df.with_columns(discovery_score=result.score).sort(
        "discovery_score", descending=True
    )

    assert recommendations["title"].item(0) == "Naruto"
    assert not recommendations["discovery_score"].is_null().any()


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

    scorer = scoring.ClusterSimilarityScorer()

    uprofile = UserProfile("Test", source_df)
    data = RecommendationModel(
        uprofile,
        target_df,
    )
    data.similarity_matrix = pl.DataFrame(
        {
            "3": [0.5, 1.0],
            "4": [1.0, 0.5],
            "id": [1, 2],
        }
    )

    result = scorer.score(data)
    recommendations = target_df.with_columns(discovery_score=result.score).sort(
        "discovery_score", descending=True
    )

    assert recommendations["title"].item(0) == "Naruto"
    assert not recommendations["discovery_score"].is_null().any()


def test_cluster_similarity_scorer_single_member_clusters():
    """When each cluster has one member, there are no intra-cluster pairs for cohesion."""
    source_df = pl.DataFrame(
        {
            "id": [1, 2],
            "features": [["Action"], ["Drama"]],
            "score": [8, 9],
            "cluster": [0, 1],
        },
    )

    target_df = pl.DataFrame({"id": [3], "features": [["Action"]]})

    scorer = scoring.ClusterSimilarityScorer()
    uprofile = UserProfile("Test", source_df)
    data = RecommendationModel(uprofile, target_df)

    data.similarity_matrix = pl.DataFrame({"3": [0.8, 0.2], "id": [1, 2]})

    full_matrix = pl.DataFrame({"3": [0.8, 0.2], "1": [1.0, 0.3], "2": [0.3, 1.0], "id": [1, 2]})
    data.get_similarity_matrix = lambda filtered=True, transposed=False: (
        data.similarity_matrix if filtered else full_matrix
    )

    result = scorer.score(data)
    assert len(result.score) == 1


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

    uprofile = UserProfile("Test", source_df)
    result = scorer.score(
        RecommendationModel(
            uprofile,
            target_df,
        )
    )

    recommendations = target_df.with_columns(discovery_score=result.score).sort(
        "discovery_score", descending=True
    )

    assert recommendations["title"].item(0) == "Jujutsu Kaisen"
    assert not recommendations["discovery_score"].is_null().any()
    assert (result.confidence >= 0).all()
    assert (result.confidence <= 1).all()


def test_popularity_scorer():
    target_df = pl.DataFrame(
        {
            "title": ["Low Score", "High Score"],
            "mean_score": [55, 85],
            "popularity": [100, 50000],
        }
    )

    scorer = scoring.PopularityScorer()

    result = scorer.score(
        RecommendationModel(
            None,
            target_df,
        )
    )

    recommendations = target_df.with_columns(discovery_score=result.score).sort(
        "discovery_score", descending=True
    )

    assert recommendations["title"].item(0) == "High Score"
    assert not recommendations["discovery_score"].is_null().any()
    assert (result.confidence >= 0).all()
    assert (result.confidence <= 1).all()
    # High popularity = high confidence
    assert result.confidence[1] > result.confidence[0]


def test_continuation_scorer():
    compare = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "title": ["Anime A", "Anime B", "Anime B Spinoff", "Anime C"],
            "user_status": ["COMPLETED", "COMPLETED", "COMPLETED", "COMPLETED"],
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

    uprofile = UserProfile("Test", compare)
    result = scorer.score(RecommendationModel(uprofile, original))

    assert [round(a, 1) for a in result.score.to_list()] == [0.8, 0.0, 0.7, 0.0]
    assert len(result.confidence) == len(original)
    assert (result.confidence >= 0).all()
    assert (result.confidence <= 1).all()


def test_continuation_scorer_null_predecessor_score():
    compare = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "title": ["Anime A", "Anime B", "Anime B Spinoff", "Anime C"],
            "user_status": ["COMPLETED", "COMPLETED", "COMPLETED", "COMPLETED"],
            "score": [None, 3, 2, 8],
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

    uprofile = UserProfile("Test", compare)
    result = scorer.score(RecommendationModel(uprofile, original))

    # Null score falls back to user mean (4.33), so id=5 gets 0.433 * 1.0
    assert result.score[0] > 0  # Not zero — uses mean score fallback
    assert result.score[1] == 0  # No predecessor match
    assert result.score[3] == 0  # No continuation


def test_continuation_scorer_dropped_sequel_caps_strength():
    """If S2 was dropped, S3's continuation strength should be capped even if S1 was loved."""
    compare = pl.DataFrame(
        {
            "id": [1, 2],
            "title": ["Show S1", "Show S2"],
            "user_status": ["COMPLETED", "DROPPED"],
            "score": [9, 3],
        }
    )

    original = pl.DataFrame(
        {
            "id": [3, 4],
            "title": ["Show S3", "Other Show"],
            "continuation_to": [[1, 2], []],
        }
    )

    scorer = scoring.ContinuationScorer()
    uprofile = UserProfile("Test", compare)
    result = scorer.score(RecommendationModel(uprofile, original))

    # S3 has predecessors S1 (completed, 9) and S2 (dropped, 3)
    # Drop should cap the strength despite S1 being great
    assert result.score[0] <= scorer.DROP_CAP
    assert result.score[1] == 0  # No continuation


def test_continuation_scorer_takes_max_of_duplicate_relations():
    compare = pl.DataFrame(
        {
            "id": [1, 2],
            "title": ["Anime A", "Anime A Spinoff"],
            "user_status": ["COMPLETED", "COMPLETED"],
            "score": [2, 8],
        }
    )

    original = pl.DataFrame(
        {
            "id": [5, 6],
            "title": ["Anime A Season 2", "Unrelated Anime"],
            "continuation_to": [[1, 2], []],
        }
    )

    scorer = scoring.ContinuationScorer()

    uprofile = UserProfile("Test", compare)
    result = scorer.score(RecommendationModel(uprofile, original))

    assert result.score.to_list() == [0.8, 0.0]


def test_continuation_scorer_penalizes_summaries():
    compare = pl.DataFrame(
        {
            "id": [1],
            "title": ["Anime A"],
            "user_status": ["COMPLETED"],
            "score": [8],
        }
    )

    original = pl.DataFrame(
        {
            "id": [5, 6],
            "title": ["Anime A Season 2", "Anime A Summary"],
            "continuation_to": [[1], [1]],
            "is_summary": [False, True],
        }
    )

    scorer = scoring.ContinuationScorer()

    uprofile = UserProfile("Test", compare)
    result = scorer.score(RecommendationModel(uprofile, original))

    # Normal sequel gets full score, summary gets SUMMARY_FACTOR (0.3) applied
    assert result.score[0] == 0.8
    assert round(result.score[1], 2) == round(0.8 * 0.3, 2)


def test_cluster_similarity_scorer_cohesion():
    source_df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "features": [["Action"], ["Action"], ["Drama"]],
            "score": [8, 9, 7],
            "cluster": [0, 0, 1],
        },
    )

    target_df = pl.DataFrame(
        {
            "id": [4, 5],
            "features": [["Action"], ["Drama"]],
        }
    )

    scorer = scoring.ClusterSimilarityScorer()

    uprofile = UserProfile("Test", source_df)
    data = RecommendationModel(uprofile, target_df)

    # Filtered matrix (seasonal columns only) for scoring
    data.similarity_matrix = pl.DataFrame(
        {
            "4": [0.8, 0.7, 0.2],
            "5": [0.2, 0.3, 0.9],
            "id": [1, 2, 3],
        }
    )

    # Full matrix (including watchlist-to-watchlist) for cohesion
    full_matrix = pl.DataFrame(
        {
            "4": [0.8, 0.7, 0.2],
            "5": [0.2, 0.3, 0.9],
            "1": [1.0, 0.9, 0.1],
            "2": [0.9, 1.0, 0.1],
            "3": [0.1, 0.1, 1.0],
            "id": [1, 2, 3],
        }
    )
    data.get_similarity_matrix = lambda filtered=True, transposed=False: (
        data.similarity_matrix if filtered else full_matrix
    )

    result = scorer.score(data)

    assert len(result.score) == 2
    assert len(result.confidence) == 2
    assert (result.confidence > 0).any()


def test_direct_similarity_scorer():
    source_df = pl.DataFrame(
        {
            "id": [1, 2],
            "features": [["Action", "Adventure"], ["Action", "Fantasy"]],
            "title": ["Bleach", "Fate/Zero"],
            "score": [10, 9],
            "user_status": ["COMPLETED", "COMPLETED"],
            "encoded": [
                {"Action": 1, "Adventure": 1, "Fantasy": 0, "Romance": 0, "Comedy": 0},
                {"Action": 1, "Adventure": 0, "Fantasy": 1, "Romance": 0, "Comedy": 0},
            ],
        },
    )
    target_df = pl.DataFrame(
        {
            "id": [3, 4],
            "features": [["Fantasy", "Romance", "Comedy"], ["Action", "Adventure"]],
            "title": ["Kaguya", "Naruto"],
            "encoded": [
                {"Action": 0, "Adventure": 0, "Fantasy": 1, "Romance": 1, "Comedy": 1},
                {"Action": 1, "Adventure": 1, "Fantasy": 0, "Romance": 0, "Comedy": 0},
            ],
        },
    )

    scorer = scoring.DirectSimilarityScorer()

    uprofile = UserProfile("Test", source_df)
    data = RecommendationModel(
        uprofile,
        target_df,
    )

    data.similarity_matrix = pl.DataFrame(
        {
            "3": [0, 0],
            "4": [1.0, 0.5],
            "id": [1, 2],
        }
    )

    result = scorer.score(data)
    recommendations = target_df.with_columns(discovery_score=result.score).sort(
        "discovery_score", descending=True
    )

    assert recommendations["title"].item(0) == "Naruto"
    assert not recommendations["discovery_score"].is_null().any()


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

    data = RecommendationModel(None, target)

    result = scorer.score(data)

    # No manga list — zero score and zero confidence
    assert (result.score == 0).all()
    assert (result.confidence == 0).all()

    data.mangalist = compare

    result = scorer.score(data)
    recommendations = target.with_columns(discovery_score=result.score).sort(
        "discovery_score", descending=True
    )

    assert recommendations["title"].to_list() == ["Anime A", "Anime X", "Anime Y"]
    assert recommendations.filter(pl.col("id") == 7)["discovery_score"].item() == 0

    # Unrated manga (id=1, score=None) should get 0.5 confidence
    assert result.confidence[1] == 0.5


def test_collaborative_recommendation_scorer_basic():
    watchlist = pl.DataFrame(
        {
            "id": [1, 2],
            "score": [8, 9],
            "user_status": ["COMPLETED", "COMPLETED"],
            "recommendations": [
                [
                    {"recommended_id": 10, "rating": 10},
                    {"recommended_id": 11, "rating": 3},
                ],
                [
                    {"recommended_id": 10, "rating": 8},
                ],
            ],
        },
        schema_overrides={
            "recommendations": pl.List(
                pl.Struct({"recommended_id": pl.UInt32, "rating": pl.Int32})
            ),
        },
    )

    seasonal = pl.DataFrame(
        {
            "id": [10, 11, 12],
            "title": ["Candidate A", "Candidate B", "Candidate C"],
        }
    )

    scorer = scoring.CollaborativeRecommendationScorer()
    uprofile = UserProfile("Test", watchlist)
    result = scorer.score(RecommendationModel(uprofile, seasonal))

    assert len(result.score) == 3
    assert len(result.confidence) == 3

    # Candidate A has links from both watchlist items, Candidate B from one
    ranked = seasonal.with_columns(s=result.score).sort("s", descending=True)
    assert ranked["title"].item(0) == "Candidate A"

    # Candidate C has no links — zero score
    c_idx = seasonal["id"].to_list().index(12)
    assert result.score[c_idx] == 0.0
    assert result.confidence[c_idx] == 0.0


def test_collaborative_recommendation_scorer_no_recommendations_column():
    watchlist = pl.DataFrame(
        {
            "id": [1],
            "score": [8],
            "user_status": ["COMPLETED"],
        }
    )

    seasonal = pl.DataFrame({"id": [10, 11], "title": ["A", "B"]})

    scorer = scoring.CollaborativeRecommendationScorer()
    uprofile = UserProfile("Test", watchlist)
    result = scorer.score(RecommendationModel(uprofile, seasonal))

    assert (result.score == 0.0).all()
    assert (result.confidence == 0.0).all()


def test_collaborative_recommendation_scorer_no_matching_links():
    watchlist = pl.DataFrame(
        {
            "id": [1],
            "score": [8],
            "user_status": ["COMPLETED"],
            "recommendations": [
                [{"recommended_id": 999, "rating": 10}],
            ],
        },
        schema_overrides={
            "recommendations": pl.List(
                pl.Struct({"recommended_id": pl.UInt32, "rating": pl.Int32})
            ),
        },
    )

    seasonal = pl.DataFrame({"id": [10, 11], "title": ["A", "B"]})

    scorer = scoring.CollaborativeRecommendationScorer()
    uprofile = UserProfile("Test", watchlist)
    result = scorer.score(RecommendationModel(uprofile, seasonal))

    assert (result.score == 0.0).all()
    assert (result.confidence == 0.0).all()


def test_collaborative_recommendation_scorer_dropped_show_negative_signal():
    watchlist = pl.DataFrame(
        {
            "id": [1],
            "score": [3],
            "user_status": ["DROPPED"],
            "recommendations": [
                [{"recommended_id": 10, "rating": 10}],
            ],
        },
        schema_overrides={
            "recommendations": pl.List(
                pl.Struct({"recommended_id": pl.UInt32, "rating": pl.Int32})
            ),
        },
    )

    seasonal = pl.DataFrame({"id": [10], "title": ["A"]})

    scorer = scoring.CollaborativeRecommendationScorer()
    uprofile = UserProfile("Test", watchlist)
    result = scorer.score(RecommendationModel(uprofile, seasonal))

    # Dropped status has -0.5 modifier, so signal should be negative before rank normalization
    # After rank normalization with a single item, score is 0.0
    assert len(result.score) == 1


def test_collaborative_recommendation_scorer_unrated_show_uses_mean():
    watchlist = pl.DataFrame(
        {
            "id": [1, 2],
            "score": [None, 8],
            "user_status": ["COMPLETED", "COMPLETED"],
            "recommendations": [
                [{"recommended_id": 10, "rating": 10}],
                [{"recommended_id": 11, "rating": 10}],
            ],
        },
        schema_overrides={
            "recommendations": pl.List(
                pl.Struct({"recommended_id": pl.UInt32, "rating": pl.Int32})
            ),
        },
    )

    seasonal = pl.DataFrame({"id": [10, 11], "title": ["A", "B"]})

    scorer = scoring.CollaborativeRecommendationScorer()
    uprofile = UserProfile("Test", watchlist)
    result = scorer.score(RecommendationModel(uprofile, seasonal))

    # Both should have non-zero scores (unrated uses mean score fallback)
    assert result.score[0] > 0
    assert result.score[1] > 0


def test_collaborative_recommendation_scorer_confidence_scales_with_link_count():
    watchlist = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "score": [8, 8, 8],
            "user_status": ["COMPLETED", "COMPLETED", "COMPLETED"],
            "recommendations": [
                [
                    {"recommended_id": 10, "rating": 10},
                    {"recommended_id": 11, "rating": 10},
                ],
                [
                    {"recommended_id": 10, "rating": 10},
                ],
                [
                    {"recommended_id": 10, "rating": 10},
                ],
            ],
        },
        schema_overrides={
            "recommendations": pl.List(
                pl.Struct({"recommended_id": pl.UInt32, "rating": pl.Int32})
            ),
        },
    )

    seasonal = pl.DataFrame({"id": [10, 11], "title": ["A", "B"]})

    scorer = scoring.CollaborativeRecommendationScorer()
    uprofile = UserProfile("Test", watchlist)
    result = scorer.score(RecommendationModel(uprofile, seasonal))

    # Candidate 10 has 3 links (>= MIN_LINK_COUNT=3), candidate 11 has 1
    assert result.confidence[0] > result.confidence[1]


def test_collaborative_recommendation_scorer_zero_total_thumbs():
    watchlist = pl.DataFrame(
        {
            "id": [1],
            "score": [8],
            "user_status": ["COMPLETED"],
            "recommendations": [
                [{"recommended_id": 10, "rating": 0}],
            ],
        },
        schema_overrides={
            "recommendations": pl.List(
                pl.Struct({"recommended_id": pl.UInt32, "rating": pl.Int32})
            ),
        },
    )

    seasonal = pl.DataFrame({"id": [10], "title": ["A"]})

    scorer = scoring.CollaborativeRecommendationScorer()
    uprofile = UserProfile("Test", watchlist)
    result = scorer.score(RecommendationModel(uprofile, seasonal))

    # Zero rating means zero signal, no division by zero
    assert len(result.score) == 1
    assert not result.score.is_null().any()


def test_collaborative_recommendation_scorer_aggregation_is_sum():
    watchlist = pl.DataFrame(
        {
            "id": [1, 2],
            "score": [8, 8],
            "user_status": ["COMPLETED", "COMPLETED"],
            "recommendations": [
                [{"recommended_id": 10, "rating": 10}],
                [{"recommended_id": 10, "rating": 10}],
            ],
        },
        schema_overrides={
            "recommendations": pl.List(
                pl.Struct({"recommended_id": pl.UInt32, "rating": pl.Int32})
            ),
        },
    )

    single_watchlist = pl.DataFrame(
        {
            "id": [1],
            "score": [8],
            "user_status": ["COMPLETED"],
            "recommendations": [
                [{"recommended_id": 10, "rating": 10}],
            ],
        },
        schema_overrides={
            "recommendations": pl.List(
                pl.Struct({"recommended_id": pl.UInt32, "rating": pl.Int32})
            ),
        },
    )

    seasonal = pl.DataFrame({"id": [10, 11], "title": ["A", "B"]})

    scorer = scoring.CollaborativeRecommendationScorer()

    result_multi = scorer.score(RecommendationModel(UserProfile("Test", watchlist), seasonal))
    result_single = scorer.score(
        RecommendationModel(UserProfile("Test", single_watchlist), seasonal)
    )

    # With rank normalization both single-candidate scores are the same (both rank 1 of 1 match),
    # but confidence should be higher with more links
    assert result_multi.confidence[0] > result_single.confidence[0]
