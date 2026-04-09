import polars as pl

from animeippo.providers.anilist.data import ALL_TAGS
from animeippo.recommendation.funnel import (
    MoodClassifier,
    _intensity_score,
    add_funnel_metadata,
)

TAG_LOOKUP = ALL_TAGS


def setup_function():
    MoodClassifier._initialized = False


def test_mood_chill():
    MoodClassifier.initialize(TAG_LOOKUP)
    ranks = [
        {"name": "Iyashikei", "rank": 90, "category": "Theme-Slice of Life"},
        {"name": "Male Protagonist", "rank": 80, "category": "Cast-Main Cast"},
    ]
    genres = ["Slice of Life"]
    moods, scores = MoodClassifier.classify(ranks, genres)
    assert "chill" in moods
    assert scores["chill"] > 0


def test_mood_hype():
    MoodClassifier.initialize(TAG_LOOKUP)
    ranks = [{"name": "Martial Arts", "rank": 85, "category": "Theme-Action"}]
    genres = ["Action"]
    moods, _ = MoodClassifier.classify(ranks, genres)
    assert "hype" in moods


def test_mood_emotional():
    MoodClassifier.initialize(TAG_LOOKUP)
    ranks = [{"name": "Tragedy", "rank": 90, "category": "Theme-Drama"}]
    genres = ["Drama"]
    moods, _ = MoodClassifier.classify(ranks, genres)
    assert "emotional" in moods


def test_mood_dark():
    MoodClassifier.initialize(TAG_LOOKUP)
    ranks = [{"name": "Gore", "rank": 80, "category": "Theme-Other"}]
    genres = ["Horror"]
    moods, _ = MoodClassifier.classify(ranks, genres)
    assert "dark" in moods


def test_mood_funny():
    MoodClassifier.initialize(TAG_LOOKUP)
    ranks = [{"name": "Parody", "rank": 70, "category": "Theme-Comedy"}]
    genres = ["Comedy"]
    moods, _ = MoodClassifier.classify(ranks, genres)
    assert "funny" in moods


def test_mood_cerebral():
    MoodClassifier.initialize(TAG_LOOKUP)
    ranks = [{"name": "Conspiracy", "rank": 85, "category": "Theme-Drama"}]
    genres = ["Mystery"]
    moods, _ = MoodClassifier.classify(ranks, genres)
    assert "cerebral" in moods


def test_mood_adventurous():
    MoodClassifier.initialize(TAG_LOOKUP)
    ranks = [{"name": "Isekai", "rank": 90, "category": "Theme-Fantasy"}]
    genres = ["Fantasy"]
    moods, _ = MoodClassifier.classify(ranks, genres)
    assert "adventurous" in moods


def test_mood_sporty():
    MoodClassifier.initialize(TAG_LOOKUP)
    ranks = [{"name": "Baseball", "rank": 90, "category": "Theme-Game-Sport"}]
    genres = ["Sports"]
    moods, _ = MoodClassifier.classify(ranks, genres)
    assert "sporty" in moods


def test_mood_multiple():
    MoodClassifier.initialize(TAG_LOOKUP)
    ranks = [
        {"name": "Gore", "rank": 90, "category": "Theme-Other"},
        {"name": "Martial Arts", "rank": 85, "category": "Theme-Action"},
    ]
    genres = ["Action", "Horror"]
    moods, _ = MoodClassifier.classify(ranks, genres)
    assert "hype" in moods
    assert "dark" in moods


def test_mood_below_threshold_excluded():
    MoodClassifier.initialize(TAG_LOOKUP)
    ranks = [{"name": "Parody", "rank": 10, "category": "Theme-Comedy"}]
    genres = []
    moods, _ = MoodClassifier.classify(ranks, genres)
    assert "funny" not in moods


def test_mood_empty_input():
    MoodClassifier.initialize(TAG_LOOKUP)
    moods, scores = MoodClassifier.classify(None, None)
    assert moods == []
    assert all(s == 0.0 for s in scores.values())


def test_mood_lazy_initialization():
    moods, _ = MoodClassifier.classify(None, None)
    assert moods == []


def test_mood_scores_returned():
    MoodClassifier.initialize(TAG_LOOKUP)
    ranks = [{"name": "Iyashikei", "rank": 90, "category": "Theme-Slice of Life"}]
    genres = ["Slice of Life", "Comedy"]
    moods, scores = MoodClassifier.classify(ranks, genres)
    assert scores["chill"] > scores["hype"]
    assert scores["funny"] > 0


def test_no_tag_in_multiple_moods():
    """Each tag in ALL_TAGS should map to at most one mood."""
    seen = {}
    for info in ALL_TAGS.values():
        mood = info.get("mood")
        if mood:
            name = info["name"]
            assert name not in seen, f"Tag '{name}' has mood in multiple entries"
            seen[name] = mood


def test_intensity_light():
    MoodClassifier.initialize(TAG_LOOKUP)
    ranks = [{"name": "Iyashikei", "rank": 90, "category": "Theme-Slice of Life"}]
    genres = ["Comedy", "Slice of Life"]
    score = _intensity_score(ranks, genres)
    assert score < 0


def test_intensity_heavy():
    MoodClassifier.initialize(TAG_LOOKUP)
    ranks = [
        {"name": "Tragedy", "rank": 90, "category": "Theme-Drama"},
        {"name": "Revenge", "rank": 80, "category": "Theme-Drama"},
    ]
    genres = ["Drama", "Psychological"]
    score = _intensity_score(ranks, genres)
    assert score > 0


def test_intensity_moderate_tags():
    MoodClassifier.initialize(TAG_LOOKUP)
    ranks = [{"name": "Coming of Age", "rank": 80, "category": "Theme-Drama"}]
    genres = ["Mystery"]
    score = _intensity_score(ranks, genres)
    assert score > 0


def test_intensity_unrecognized_tag_ignored():
    MoodClassifier.initialize(TAG_LOOKUP)
    ranks = [{"name": "Male Protagonist", "rank": 90, "category": "Cast-Main Cast"}]
    score = _intensity_score(ranks, None)
    assert score == 0.0


def test_intensity_empty_input():
    MoodClassifier.initialize(TAG_LOOKUP)
    assert _intensity_score(None, None) == 0.0


def test_bucket_intensity_chill_only_override():
    from animeippo.recommendation.funnel import _bucket_intensity

    scores = [0.0, -2.0, 1.0]
    moods = [["hype"], ["chill"], ["dark"]]
    labels = _bucket_intensity(scores, moods)
    assert labels[1] == "light"


def test_bucket_intensity_all_overrides():
    from animeippo.recommendation.funnel import _bucket_intensity

    scores = [-2.0, 5.0]
    moods = [["chill"], ["dark"]]
    labels = _bucket_intensity(scores, moods)
    assert labels[0] == "light"
    assert labels[1] == "all_in"


def test_bucket_intensity_percentile_thirds():
    from animeippo.recommendation.funnel import _bucket_intensity

    scores = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 8.0]
    moods = [["funny"]] * 9
    labels = _bucket_intensity(scores, moods)
    assert "light" in labels
    assert "moderate" in labels
    assert "all_in" in labels


def test_add_funnel_metadata():
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "title": ["Chill Show", "Dark Show"],
            "genres": [["Slice of Life", "Comedy"], ["Horror", "Thriller"]],
            "temp_ranks": [
                [{"name": "Iyashikei", "rank": 90, "category": "Theme-Slice of Life"}],
                [
                    {"name": "Gore", "rank": 85, "category": "Theme-Other"},
                    {"name": "Torture", "rank": 70, "category": "Theme-Other"},
                ],
            ],
        }
    )

    result = add_funnel_metadata(df, tag_lookup=TAG_LOOKUP)

    assert "moods" in result.columns
    assert "intensity" in result.columns
    assert "intensity_score" in result.columns
    assert "mood_chill" in result.columns
    assert "mood_dark" in result.columns

    assert "chill" in result["moods"][0]
    assert "dark" in result["moods"][1]

    assert result["mood_chill"][0] > 0
    assert result["mood_dark"][1] > 0


def test_unknown_tags_logged(caplog):
    df = pl.DataFrame(
        {
            "id": [1],
            "title": ["Show"],
            "genres": [["Action"]],
            "temp_ranks": [
                [{"name": "FakeTag9999", "rank": 50, "category": "Theme-Other"}],
            ],
        }
    )

    tag_lookup = {1: {"name": "Gore", "category": "Theme-Other", "isAdult": False}}

    with caplog.at_level("DEBUG"):
        add_funnel_metadata(df, tag_lookup=tag_lookup)

    assert "FakeTag9999" in caplog.text


def test_add_funnel_metadata_without_temp_ranks():
    df = pl.DataFrame(
        {
            "id": [1],
            "title": ["Action Show"],
            "genres": [["Action", "Mecha"]],
        }
    )

    result = add_funnel_metadata(df, tag_lookup=TAG_LOOKUP)

    assert "moods" in result.columns
    assert "intensity" in result.columns
    assert "hype" in result["moods"][0]
