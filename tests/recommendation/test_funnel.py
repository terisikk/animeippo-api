import polars as pl

from animeippo.recommendation.funnel import (
    MoodClassifier,
    add_funnel_metadata,
    classify_intensity,
)


def test_mood_chill():
    ranks = [
        {"name": "Iyashikei", "rank": 90, "category": "Theme-Slice of Life"},
        {"name": "Male Protagonist", "rank": 80, "category": "Cast-Main Cast"},
    ]
    genres = ["Slice of Life"]
    moods = MoodClassifier.classify(ranks, genres)
    assert "chill" in moods


def test_mood_hype():
    ranks = [{"name": "Martial Arts", "rank": 85, "category": "Theme-Action"}]
    genres = ["Action"]
    moods = MoodClassifier.classify(ranks, genres)
    assert "hype" in moods


def test_mood_emotional():
    ranks = [{"name": "Tragedy", "rank": 90, "category": "Theme-Drama"}]
    genres = ["Drama"]
    moods = MoodClassifier.classify(ranks, genres)
    assert "emotional" in moods


def test_mood_dark():
    ranks = [{"name": "Gore", "rank": 80, "category": "Theme-Other"}]
    genres = ["Horror"]
    moods = MoodClassifier.classify(ranks, genres)
    assert "dark" in moods


def test_mood_funny():
    ranks = [{"name": "Parody", "rank": 70, "category": "Theme-Comedy"}]
    genres = ["Comedy"]
    moods = MoodClassifier.classify(ranks, genres)
    assert "funny" in moods


def test_mood_multiple():
    ranks = [
        {"name": "Gore", "rank": 90, "category": "Theme-Other"},
        {"name": "Martial Arts", "rank": 85, "category": "Theme-Action"},
    ]
    genres = ["Action", "Horror"]
    moods = MoodClassifier.classify(ranks, genres)
    assert "hype" in moods
    assert "dark" in moods


def test_mood_below_threshold_excluded():
    ranks = [{"name": "Parody", "rank": 10, "category": "Theme-Comedy"}]
    genres = []
    moods = MoodClassifier.classify(ranks, genres)
    # rank 10 / 100 = 0.1, below threshold 1.0
    assert "funny" not in moods


def test_mood_empty_input():
    moods = MoodClassifier.classify(None, None)
    assert moods == []


def test_intensity_light():
    ranks = [{"name": "Iyashikei", "rank": 90, "category": "Theme-Slice of Life"}]
    genres = ["Comedy", "Slice of Life"]
    assert classify_intensity(ranks, genres) == "light"


def test_intensity_all_in():
    ranks = [
        {"name": "Tragedy", "rank": 90, "category": "Theme-Drama"},
        {"name": "Revenge", "rank": 80, "category": "Theme-Drama"},
    ]
    genres = ["Drama", "Psychological"]
    assert classify_intensity(ranks, genres) == "all_in"


def test_intensity_moderate():
    ranks = [{"name": "Coming of Age", "rank": 70, "category": "Theme-Drama"}]
    genres = ["Drama", "Comedy"]
    assert classify_intensity(ranks, genres) == "moderate"


def test_intensity_empty_input():
    assert classify_intensity(None, None) == "moderate"


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

    result = add_funnel_metadata(df)

    assert "moods" in result.columns
    assert "intensity" in result.columns

    assert "chill" in result["moods"][0]
    assert "dark" in result["moods"][1]

    assert result["intensity"][0] == "light"
    assert result["intensity"][1] == "all_in"


def test_add_funnel_metadata_without_temp_ranks():
    df = pl.DataFrame(
        {
            "id": [1],
            "title": ["Action Show"],
            "genres": [["Action", "Mecha"]],
        }
    )

    result = add_funnel_metadata(df)

    assert "moods" in result.columns
    assert "intensity" in result.columns
    # Action + Mecha genres alone = 2.0 hype score, above threshold
    assert "hype" in result["moods"][0]
