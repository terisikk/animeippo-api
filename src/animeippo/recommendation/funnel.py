"""Mood and intensity classification for the recommendation funnel.

Adds metadata columns that the frontend uses for progressive filtering
and that the category system can use for mood-based categories.
"""

from typing import ClassVar

import polars as pl

MOOD_THRESHOLD = 1.0


class MoodClassifier:
    """Classifies anime into moods based on tags and genres."""

    MOOD_TAGS: ClassVar[dict[str, list[str]]] = {
        "chill": [
            "Iyashikei",
            "Cute Girls Doing Cute Things",
            "Cute Boys Doing Cute Things",
            "Family Life",
            "Agriculture",
            "Horticulture",
            "Food",
            "Camping",
            "Outdoor Activities",
        ],
        "hype": [
            "Martial Arts",
            "Swordplay",
            "Battle Royale",
            "Super Power",
            "Superhero",
            "Kaiju",
            "Guns",
            "Espionage",
            "Fugitive",
        ],
        "emotional": [
            "Tragedy",
            "Coming of Age",
            "Unrequited Love",
            "Love Triangle",
            "Rehabilitation",
            "Found Family",
            "Bullying",
            "Suicide",
            "Disability",
        ],
        "dark": [
            "Gore",
            "Cosmic Horror",
            "Body Horror",
            "Denpa",
            "Torture",
            "Death Game",
            "Cannibalism",
            "Slavery",
            "Drugs",
            "Ero Guro",
            "Survival",
            "Noir",
        ],
        "funny": [
            "Parody",
            "Satire",
            "Slapstick",
            "Surreal Comedy",
        ],
    }

    MOOD_GENRES: ClassVar[dict[str, list[str]]] = {
        "chill": ["Slice of Life"],
        "hype": ["Action", "Mecha"],
        "emotional": ["Drama", "Romance"],
        "dark": ["Horror", "Thriller", "Psychological"],
        "funny": ["Comedy"],
    }

    # Tags that build up per-mood lookup: {tag_name: mood}
    _tag_to_mood: ClassVar[dict[str, str]] = {
        tag: mood for mood, tags in MOOD_TAGS.items() for tag in tags
    }

    _genre_to_moods: ClassVar[dict[str, list[str]]] = {}

    @classmethod
    def _build_genre_lookup(cls):
        if not cls._genre_to_moods:
            for mood, genres in cls.MOOD_GENRES.items():
                for genre in genres:
                    cls._genre_to_moods.setdefault(genre, []).append(mood)

    @classmethod
    def classify(cls, temp_ranks, genres):
        """Compute mood scores and return list of moods above threshold.

        Args:
            temp_ranks: list of {name, rank, category} dicts
            genres: list of genre strings
        """
        cls._build_genre_lookup()

        scores = dict.fromkeys(cls.MOOD_TAGS, 0.0)

        if temp_ranks:
            for tag in temp_ranks:
                mood = cls._tag_to_mood.get(tag["name"])
                if mood:
                    scores[mood] += tag["rank"] / 100.0

        if genres:
            for genre in genres:
                for mood in cls._genre_to_moods.get(genre, []):
                    scores[mood] += 1.0

        return [mood for mood, score in scores.items() if score >= MOOD_THRESHOLD]


ALL_IN_THRESHOLD = 1.5
LIGHT_THRESHOLD = -0.5

# Genres/tags that push toward heavy or light intensity
INTENSITY_WEIGHTS = {
    "heavy_genres": {"Drama", "Psychological", "Thriller", "Horror"},
    "light_genres": {"Comedy", "Slice of Life", "Music"},
    "heavy_tags": {
        "Conspiracy",
        "Revenge",
        "Suicide",
        "Tragedy",
        "Death Game",
        "Gore",
        "Cosmic Horror",
        "Denpa",
        "Torture",
        "Body Horror",
        "War",
        "Military",
        "Slavery",
        "Cannibalism",
        "Drugs",
        "Noir",
        "Ero Guro",
        "Philosophy",
    },
    "light_tags": {
        "Iyashikei",
        "Parody",
        "Satire",
        "Slapstick",
        "Surreal Comedy",
        "Chibi",
        "Cute Girls Doing Cute Things",
        "Cute Boys Doing Cute Things",
        "Food",
        "Camping",
        "Outdoor Activities",
    },
}


def _intensity_score(temp_ranks, genres):
    score = 0.0
    if genres:
        for genre in genres:
            if genre in INTENSITY_WEIGHTS["heavy_genres"]:
                score += 1.0
            elif genre in INTENSITY_WEIGHTS["light_genres"]:
                score -= 1.0
    if temp_ranks:
        for tag in temp_ranks:
            rank = tag["rank"] / 100.0
            if tag["name"] in INTENSITY_WEIGHTS["heavy_tags"]:
                score += rank
            elif tag["name"] in INTENSITY_WEIGHTS["light_tags"]:
                score -= rank
    return score


def classify_intensity(temp_ranks, genres):
    """Classify narrative depth as light/moderate/all_in."""
    score = _intensity_score(temp_ranks, genres)

    if score >= ALL_IN_THRESHOLD:
        return "all_in"
    if score <= LIGHT_THRESHOLD:
        return "light"
    return "moderate"


def add_funnel_metadata(recommendations):
    """Add mood and intensity columns to recommendations DataFrame."""
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
    intensity_list = []

    for ranks, genres in zip(ranks_data, genres_data, strict=True):
        moods_list.append(MoodClassifier.classify(ranks, genres))
        intensity_list.append(classify_intensity(ranks, genres))

    return recommendations.with_columns(
        moods=pl.Series(moods_list, dtype=pl.List(pl.Utf8)),
        intensity=pl.Series(intensity_list, dtype=pl.Utf8),
    )
