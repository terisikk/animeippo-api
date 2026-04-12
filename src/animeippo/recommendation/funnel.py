"""Mood and intensity classification for the recommendation funnel.

Adds metadata columns that the frontend uses for progressive filtering
and that the category system can use for mood-based categories.
"""

import polars as pl

from animeippo.providers.anilist.data import ALL_GENRES

MOOD_THRESHOLD = 1.0
GENRE_WEIGHT = 1.0
MOOD_TAG_RANK_CUTOFF = 60

TIER_SIGNS = {"heavy": 1.0, "light": -1.0, "moderate": 0.5}

MOOD_NAMES = ["chill", "hype", "emotional", "dark", "funny", "cerebral", "adventurous", "sporty"]

# Override threshold for dark-only shows
DARK_ONLY_OVERRIDE_THRESHOLD = 3.0


def classify_mood(temp_ranks, genres):
    """Compute mood scores and return moods above threshold."""
    scores = dict.fromkeys(MOOD_NAMES, 0.0)

    for tag in temp_ranks:
        mood = tag.get("mood")
        if tag["rank"] >= MOOD_TAG_RANK_CUTOFF and mood:
            scores[mood] += tag["rank"] / 100.0

    for genre in genres:
        mood = ALL_GENRES.get(genre, {}).get("mood")
        if mood:
            scores[mood] += GENRE_WEIGHT

    return [mood for mood, score in scores.items() if score >= MOOD_THRESHOLD]


def classify_intensity(temp_ranks, genres):
    """Compute raw intensity score from tags and genres."""
    score = 0.0

    for genre in genres:
        score += TIER_SIGNS.get(ALL_GENRES.get(genre, {}).get("intensity"), 0.0)

    for tag in temp_ranks:
        score += tag["rank"] / 100.0 * TIER_SIGNS.get(tag.get("intensity"), 0.0)

    return score


def _bucket_intensity(scores, moods_list):
    """Bucket intensity scores using percentile thirds.

    Chill-only shows are auto-assigned light; dark-only high-score shows
    are auto-assigned all_in. Remaining shows are split into percentile thirds.
    """
    results = [None] * len(scores)
    remaining_indices = []

    for i, (score, moods) in enumerate(zip(scores, moods_list, strict=True)):
        if moods == ["chill"]:
            results[i] = "light"
        elif moods == ["dark"] and score > DARK_ONLY_OVERRIDE_THRESHOLD:
            results[i] = "all_in"
        else:
            remaining_indices.append(i)

    if remaining_indices:
        remaining_scores = sorted(scores[i] for i in remaining_indices)
        n = len(remaining_scores)
        low_cutoff = remaining_scores[n // 3]
        high_cutoff = remaining_scores[2 * n // 3]

        for i in remaining_indices:
            if scores[i] <= low_cutoff:
                results[i] = "light"
            elif scores[i] >= high_cutoff:
                results[i] = "all_in"
            else:
                results[i] = "moderate"

    return results


def add_funnel_metadata(recommendations):
    """Add mood and intensity columns to recommendations DataFrame."""
    moods_list = []
    intensity_scores = []

    for row in recommendations.iter_rows(named=True):
        ranks = row.get("temp_ranks", [])
        genres = row.get("genres", [])

        moods = classify_mood(ranks, genres)
        moods_list.append(moods)
        intensity_scores.append(classify_intensity(ranks, genres))

    intensity_labels = _bucket_intensity(intensity_scores, moods_list)

    return recommendations.with_columns(
        moods=pl.Series(moods_list, dtype=pl.List(pl.Utf8)),
        intensity=pl.Series(intensity_labels, dtype=pl.Utf8),
        intensity_score=pl.Series(intensity_scores, dtype=pl.Float64),
    )
