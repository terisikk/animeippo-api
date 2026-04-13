import polars as pl

from animeippo.providers.anilist.data import ALL_GENRES, ALL_TAGS, TAG_BY_NAME
from animeippo.recommendation.funnel import (
    _bucket_intensity,
    add_funnel_metadata,
)


def _tag(name, rank):
    """Build a feature info struct for a tag."""
    info = TAG_BY_NAME.get(name, {})
    return {
        "name": name,
        "rank": rank,
        "category": info.get("category", ""),
        "mood": info.get("mood"),
        "intensity": info.get("intensity"),
    }


def _genre(name):
    """Build a feature info struct for a genre."""
    info = ALL_GENRES.get(name, {})
    return {
        "name": name,
        "rank": 100,
        "category": "Genre",
        "mood": info.get("mood"),
        "intensity": info.get("intensity"),
    }


def _make_df(feature_info):
    """Build a minimal DataFrame with feature_info for testing."""
    return pl.DataFrame({"id": list(range(len(feature_info))), "feature_info": feature_info})


def test_mood_chill():
    result = add_funnel_metadata(_make_df([[_tag("Iyashikei", 90), _genre("Slice of Life")]]))
    assert "chill" in result["moods"][0]


def test_mood_hype():
    result = add_funnel_metadata(_make_df([[_tag("Martial Arts", 85), _genre("Action")]]))
    assert "hype" in result["moods"][0]


def test_mood_emotional():
    result = add_funnel_metadata(_make_df([[_tag("Tragedy", 90), _genre("Drama")]]))
    assert "emotional" in result["moods"][0]


def test_mood_dark():
    result = add_funnel_metadata(_make_df([[_tag("Gore", 80), _genre("Horror")]]))
    assert "dark" in result["moods"][0]


def test_mood_funny():
    result = add_funnel_metadata(_make_df([[_tag("Parody", 70), _genre("Comedy")]]))
    assert "funny" in result["moods"][0]


def test_mood_cerebral():
    result = add_funnel_metadata(_make_df([[_tag("Conspiracy", 85), _genre("Mystery")]]))
    assert "cerebral" in result["moods"][0]


def test_mood_adventurous():
    result = add_funnel_metadata(_make_df([[_tag("Isekai", 90), _genre("Fantasy")]]))
    assert "adventurous" in result["moods"][0]


def test_mood_sporty():
    result = add_funnel_metadata(_make_df([[_tag("Baseball", 90), _genre("Sports")]]))
    assert "sporty" in result["moods"][0]


def test_mood_multiple():
    features = [_tag("Gore", 90), _tag("Martial Arts", 85), _genre("Action"), _genre("Horror")]
    result = add_funnel_metadata(_make_df([features]))
    moods = result["moods"][0]
    assert "hype" in moods
    assert "dark" in moods


def test_mood_below_threshold_excluded():
    result = add_funnel_metadata(_make_df([[_tag("Parody", 10)]]))
    assert "funny" not in result["moods"][0]


def test_mood_genre_without_mood_ignored():
    """Genres like Supernatural have no mood, should not contribute to mood scoring."""
    result = add_funnel_metadata(_make_df([[_genre("Supernatural"), _genre("Action")]]))
    assert "hype" in result["moods"][0]


def test_mood_tag_without_mood_ignored():
    """Tags without mood (like cast tags) should not contribute to mood scoring."""
    result = add_funnel_metadata(_make_df([[_tag("Male Protagonist", 90), _genre("Action")]]))
    assert result["moods"][0].to_list() == ["hype"]


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
    result = add_funnel_metadata(
        _make_df([[_tag("Iyashikei", 90), _genre("Comedy"), _genre("Slice of Life")]])
    )
    assert result["intensity_score"][0] < 0


def test_intensity_heavy():
    features = [_tag("Tragedy", 90), _tag("Revenge", 80), _genre("Drama"), _genre("Psychological")]
    result = add_funnel_metadata(_make_df([features]))
    assert result["intensity_score"][0] > 0


def test_intensity_moderate_tags():
    result = add_funnel_metadata(_make_df([[_tag("Coming of Age", 80), _genre("Mystery")]]))
    assert result["intensity_score"][0] > 0


def test_intensity_tag_without_intensity_ignored():
    """Tags without intensity contribute 0 to intensity score."""
    result = add_funnel_metadata(_make_df([[_tag("Male Protagonist", 90), _genre("Supernatural")]]))
    assert result["intensity_score"][0] == 0.0


def test_bucket_intensity_percentile_thirds():
    scores = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 8.0]
    df = pl.DataFrame({"intensity_score": scores})
    labels = _bucket_intensity(df)["intensity"].to_list()
    assert "light" in labels
    assert "moderate" in labels
    assert "all_in" in labels


def test_add_funnel_metadata():
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "title": ["Chill Show", "Dark Show"],
            "feature_info": [
                [_tag("Iyashikei", 90), _genre("Slice of Life"), _genre("Comedy")],
                [_tag("Gore", 85), _tag("Torture", 70), _genre("Horror"), _genre("Thriller")],
            ],
        }
    )

    result = add_funnel_metadata(df)

    assert "moods" in result.columns
    assert "intensity" in result.columns
    assert "intensity_score" in result.columns
    assert "chill" in result["moods"][0]
    assert "dark" in result["moods"][1]
