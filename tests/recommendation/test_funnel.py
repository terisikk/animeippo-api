import polars as pl

from animeippo.providers.anilist.data import ALL_TAGS, TAG_BY_NAME
from animeippo.recommendation.funnel import (
    _bucket_intensity,
    add_funnel_metadata,
    classify_intensity,
    classify_mood,
)


def _tag(name, rank):
    """Build a tag struct with mood/intensity from TAG_BY_NAME."""
    info = TAG_BY_NAME.get(name, {})
    return {
        "name": name,
        "rank": rank,
        "category": info.get("category", ""),
        "mood": info.get("mood"),
        "intensity": info.get("intensity"),
    }


def test_mood_chill():
    ranks = [_tag("Iyashikei", 90), _tag("Male Protagonist", 80)]
    assert "chill" in classify_mood(ranks, ["Slice of Life"])


def test_mood_hype():
    assert "hype" in classify_mood([_tag("Martial Arts", 85)], ["Action"])


def test_mood_emotional():
    assert "emotional" in classify_mood([_tag("Tragedy", 90)], ["Drama"])


def test_mood_dark():
    assert "dark" in classify_mood([_tag("Gore", 80)], ["Horror"])


def test_mood_funny():
    assert "funny" in classify_mood([_tag("Parody", 70)], ["Comedy"])


def test_mood_cerebral():
    assert "cerebral" in classify_mood([_tag("Conspiracy", 85)], ["Mystery"])


def test_mood_adventurous():
    assert "adventurous" in classify_mood([_tag("Isekai", 90)], ["Fantasy"])


def test_mood_sporty():
    assert "sporty" in classify_mood([_tag("Baseball", 90)], ["Sports"])


def test_mood_multiple():
    ranks = [_tag("Gore", 90), _tag("Martial Arts", 85)]
    moods = classify_mood(ranks, ["Action", "Horror"])
    assert "hype" in moods
    assert "dark" in moods


def test_mood_below_threshold_excluded():
    assert "funny" not in classify_mood([_tag("Parody", 10)], [])


def test_mood_genre_without_mood_ignored():
    """Genres like Supernatural have no mood mapping and should not contribute."""
    assert classify_mood([], ["Supernatural"]) == []


def test_mood_tag_without_mood_ignored():
    assert classify_mood([_tag("Male Protagonist", 90)], []) == []


def test_mood_empty_input():
    assert classify_mood([], []) == []


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
    score = classify_intensity([_tag("Iyashikei", 90)], ["Comedy", "Slice of Life"])
    assert score < 0


def test_intensity_heavy():
    ranks = [_tag("Tragedy", 90), _tag("Revenge", 80)]
    score = classify_intensity(ranks, ["Drama", "Psychological"])
    assert score > 0


def test_intensity_moderate_tags():
    score = classify_intensity([_tag("Coming of Age", 80)], ["Mystery"])
    assert score > 0


def test_intensity_tag_without_intensity_ignored():
    score = classify_intensity([_tag("Male Protagonist", 90)], [])
    assert score == 0.0


def test_intensity_empty_input():
    assert classify_intensity([], []) == 0.0


def test_bucket_intensity_chill_only_override():
    scores = [0.0, -2.0, 1.0]
    moods = [["hype"], ["chill"], ["dark"]]
    labels = _bucket_intensity(scores, moods)
    assert labels[1] == "light"


def test_bucket_intensity_all_overrides():
    scores = [-2.0, 5.0]
    moods = [["chill"], ["dark"]]
    labels = _bucket_intensity(scores, moods)
    assert labels[0] == "light"
    assert labels[1] == "all_in"


def test_bucket_intensity_percentile_thirds():
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
                [_tag("Iyashikei", 90)],
                [_tag("Gore", 85), _tag("Torture", 70)],
            ],
        }
    )

    result = add_funnel_metadata(df)

    assert "moods" in result.columns
    assert "intensity" in result.columns
    assert "intensity_score" in result.columns
    assert "chill" in result["moods"][0]
    assert "dark" in result["moods"][1]


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
    assert "hype" in result["moods"][0]
