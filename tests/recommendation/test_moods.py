import polars as pl

from animeippo.providers.anilist.data import ALL_GENRES, ALL_TAGS, TAG_BY_NAME
from animeippo.recommendation.moods import (
    add_mood_data,
    bucket_intensity,
    compute_intensity,
    explode_features,
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
    result = add_mood_data(_make_df([[_tag("Iyashikei", 90), _genre("Slice of Life")]]))

    mood_names = result.select(pl.col("moods").list.eval(pl.element().struct.field("mood")))

    assert "chill" in mood_names[0, "moods"]


def test_mood_hype():
    result = add_mood_data(_make_df([[_tag("Martial Arts", 85), _genre("Action")]]))

    mood_names = result.select(pl.col("moods").list.eval(pl.element().struct.field("mood")))

    assert "hype" in mood_names[0, "moods"]


def test_mood_emotional():
    result = add_mood_data(_make_df([[_tag("Tragedy", 90), _genre("Drama")]]))
    mood_names = result.select(pl.col("moods").list.eval(pl.element().struct.field("mood")))
    assert "emotional" in mood_names[0, "moods"]


def test_mood_dark():
    result = add_mood_data(_make_df([[_tag("Gore", 80), _genre("Horror")]]))
    mood_names = result.select(pl.col("moods").list.eval(pl.element().struct.field("mood")))
    assert "dark" in mood_names[0, "moods"]


def test_mood_funny():
    result = add_mood_data(_make_df([[_tag("Parody", 70), _genre("Comedy")]]))
    mood_names = result.select(pl.col("moods").list.eval(pl.element().struct.field("mood")))
    assert "funny" in mood_names[0, "moods"]


def test_mood_cerebral():
    result = add_mood_data(_make_df([[_tag("Conspiracy", 85), _genre("Mystery")]]))
    mood_names = result.select(pl.col("moods").list.eval(pl.element().struct.field("mood")))
    assert "cerebral" in mood_names[0, "moods"]


def test_mood_adventurous():
    result = add_mood_data(_make_df([[_tag("Isekai", 90), _genre("Fantasy")]]))
    mood_names = result.select(pl.col("moods").list.eval(pl.element().struct.field("mood")))
    assert "adventurous" in mood_names[0, "moods"]


def test_mood_sporty():
    result = add_mood_data(_make_df([[_tag("Baseball", 90), _genre("Sports")]]))
    mood_names = result.select(pl.col("moods").list.eval(pl.element().struct.field("mood")))
    assert "sporty" in mood_names[0, "moods"]


def test_mood_multiple():
    features = [_tag("Gore", 90), _tag("Martial Arts", 85), _genre("Action"), _genre("Horror")]
    result = add_mood_data(_make_df([features]))
    mood_names = result.select(pl.col("moods").list.eval(pl.element().struct.field("mood")))
    moods = mood_names[0, "moods"]
    assert "hype" in moods
    assert "dark" in moods


def test_mood_below_threshold_excluded():
    result = add_mood_data(_make_df([[_tag("Parody", 10)]]))
    mood_names = result.select(pl.col("moods").list.eval(pl.element().struct.field("mood")))
    assert mood_names[0, "moods"] is None


def test_mood_genre_without_mood_ignored():
    """Genres like Supernatural have no mood, should not contribute to mood scoring."""
    result = add_mood_data(_make_df([[_genre("Supernatural"), _genre("Action")]]))
    mood_names = result.select(pl.col("moods").list.eval(pl.element().struct.field("mood")))
    assert "hype" in mood_names[0, "moods"]


def test_mood_tag_without_mood_ignored():
    """Tags without mood (like cast tags) should not contribute to mood scoring."""
    result = add_mood_data(_make_df([[_tag("Male Protagonist", 90), _genre("Action")]]))
    mood_names = result.select(pl.col("moods").list.eval(pl.element().struct.field("mood")))
    assert mood_names[0, "moods"].to_list() == ["hype"]


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
    features = explode_features(
        _make_df([[_tag("Iyashikei", 90), _genre("Comedy"), _genre("Slice of Life")]])
    )
    result = compute_intensity(features)
    assert result["intensity_score"][0] < 0


def test_intensity_heavy():
    features = explode_features(
        _make_df(
            [[_tag("Tragedy", 90), _tag("Revenge", 80), _genre("Drama"), _genre("Psychological")]]
        )
    )
    result = compute_intensity(features)
    assert result["intensity_score"][0] > 0


def test_intensity_moderate_tags():
    features = explode_features(_make_df([[_tag("Coming of Age", 80), _genre("Mystery")]]))
    result = compute_intensity(features)
    assert result["intensity_score"][0] > 0


def test_intensity_tag_without_intensity_ignored():
    """Tags without intensity contribute 0 to intensity score."""
    features = explode_features(_make_df([[_tag("Male Protagonist", 90), _genre("Supernatural")]]))
    result = compute_intensity(features)
    assert result["intensity_score"][0] == 0.0


def test_bucket_intensity_percentile_thirds():
    scores = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 8.0]
    df = pl.DataFrame({"intensity_score": scores, "id": 1})
    labels = bucket_intensity(df)["intensity"].to_list()
    assert "light" in labels
    assert "moderate" in labels
    assert "all_in" in labels


def test_add_mood_data():
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

    result = add_mood_data(df)

    assert "moods" in result.columns
    assert "intensity" in result.columns
    mood_names = result.select(pl.col("moods").list.eval(pl.element().struct.field("mood")))
    assert "chill" in mood_names[0, "moods"]
    assert "dark" in mood_names[1, "moods"]
