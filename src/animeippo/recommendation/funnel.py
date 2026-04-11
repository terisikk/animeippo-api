"""Mood and intensity classification for the recommendation funnel.

Adds metadata columns that the frontend uses for progressive filtering
and that the category system can use for mood-based categories.
"""

from typing import ClassVar

import polars as pl
import structlog

logger = structlog.get_logger()

MOOD_THRESHOLD = 1.0
GENRE_WEIGHT = 1.0
MOOD_TAG_RANK_CUTOFF = 60

# Genre-to-mood mapping (tags get their mood from data.py)
MOOD_GENRES: dict[str, list[str]] = {
    "chill": ["Slice of Life"],
    "hype": ["Action", "Mecha"],
    "emotional": ["Drama", "Music", "Romance"],
    "dark": ["Horror", "Thriller"],
    "funny": ["Comedy"],
    "cerebral": ["Mystery", "Psychological"],
    "adventurous": ["Adventure", "Fantasy", "Sci-Fi"],
    "sporty": ["Sports"],
}

# Intensity genre tiers
INTENSITY_GENRE_TIERS: dict[str, str] = {
    "Psychological": "heavy",
    "Thriller": "heavy",
    "Horror": "heavy",
    "Drama": "heavy",
    "Slice of Life": "light",
    "Comedy": "light",
    "Mystery": "moderate",
}

TIER_SIGNS = {"heavy": 1.0, "light": -1.0, "moderate": 0.5}

# Override threshold for dark-only shows
DARK_ONLY_OVERRIDE_THRESHOLD = 3.0


class MoodClassifier:
    """Classifies anime into moods based on tags and genres."""

    _tag_to_mood: ClassVar[dict[str, str]] = {}
    _genre_to_moods: ClassVar[dict[str, list[str]]] = {}
    _mood_names: ClassVar[list[str]] = []
    _initialized: ClassVar[bool] = False

    @classmethod
    def initialize(cls, tag_lookup=None):
        """Build lookup tables from tag_lookup data."""
        cls._tag_to_mood = {}
        if tag_lookup:
            for info in tag_lookup.values():
                mood = info.get("mood")
                if mood:
                    cls._tag_to_mood[info["name"]] = mood

        cls._genre_to_moods = {}
        for mood, genres in MOOD_GENRES.items():
            for genre in genres:
                cls._genre_to_moods.setdefault(genre, []).append(mood)

        cls._mood_names = list(MOOD_GENRES.keys())
        cls._initialized = True

    @classmethod
    def classify(cls, temp_ranks, genres):
        """Compute mood scores and return moods above threshold with raw scores."""
        if not cls._initialized:
            cls.initialize()

        scores = dict.fromkeys(cls._mood_names, 0.0)

        if temp_ranks:
            for tag in temp_ranks:
                if tag["rank"] < MOOD_TAG_RANK_CUTOFF:
                    continue
                mood = cls._tag_to_mood.get(tag["name"])
                if mood:
                    scores[mood] += tag["rank"] / 100.0

        if genres:
            for genre in genres:
                for mood in cls._genre_to_moods.get(genre, []):
                    scores[mood] += GENRE_WEIGHT

        moods = [mood for mood, score in scores.items() if score >= MOOD_THRESHOLD]
        return moods, scores


# Intensity tag tier lookup built from tag_lookup data
_intensity_tag_tiers: dict[str, str] = {}


def _initialize_intensity(tag_lookup):
    global _intensity_tag_tiers  # noqa: PLW0603
    _intensity_tag_tiers = {}
    if tag_lookup:
        for info in tag_lookup.values():
            intensity = info.get("intensity")
            if intensity:
                _intensity_tag_tiers[info["name"]] = intensity


def _intensity_score(temp_ranks, genres):
    score = 0.0

    if genres:
        for genre in genres:
            tier = INTENSITY_GENRE_TIERS.get(genre)
            if tier:
                score += TIER_SIGNS[tier]

    if temp_ranks:
        for tag in temp_ranks:
            tier = _intensity_tag_tiers.get(tag["name"])
            if tier:
                score += tag["rank"] / 100.0 * TIER_SIGNS[tier]

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


def _log_unknown_tags(recommendations, tag_lookup):
    """Log warning if anime have tags not in the static tag lookup."""
    if tag_lookup is None or "temp_ranks" not in recommendations.columns:
        return

    known_names = {info["name"] for info in tag_lookup.values()}
    all_tag_names = set(
        recommendations["temp_ranks"].explode().struct.field("name").drop_nulls().unique().to_list()
    )
    unknown = all_tag_names - known_names
    if unknown:
        logger.warning("unknown_tags", count=len(unknown), tags=sorted(unknown))


def add_funnel_metadata(recommendations, tag_lookup=None):
    """Add mood, mood scores, and intensity columns to recommendations DataFrame."""
    MoodClassifier.initialize(tag_lookup)
    _initialize_intensity(tag_lookup)
    _log_unknown_tags(recommendations, tag_lookup)

    ranks_data = (
        recommendations["temp_ranks"].to_list()
        if "temp_ranks" in recommendations.columns
        else [None] * len(recommendations)
    )
    genres_data = (
        recommendations["genres"].to_list()
        if "genres" in recommendations.columns
        else [None] * len(recommendations)
    )

    moods_list = []
    mood_scores_list = []
    intensity_scores = []

    for ranks, genres in zip(ranks_data, genres_data, strict=True):
        moods, scores = MoodClassifier.classify(ranks, genres)
        moods_list.append(moods)
        mood_scores_list.append(scores)
        intensity_scores.append(_intensity_score(ranks, genres))

    intensity_labels = _bucket_intensity(intensity_scores, moods_list)

    mood_names = MoodClassifier._mood_names
    mood_score_columns = {
        f"mood_{mood}": pl.Series([scores[mood] for scores in mood_scores_list], dtype=pl.Float64)
        for mood in mood_names
    }

    return recommendations.with_columns(
        moods=pl.Series(moods_list, dtype=pl.List(pl.Utf8)),
        intensity=pl.Series(intensity_labels, dtype=pl.Utf8),
        intensity_score=pl.Series(intensity_scores, dtype=pl.Float64),
        **mood_score_columns,
    )
