import polars as pl

from animeippo.clustering.namer import ClusterNamer


def test_cluster_name_generation_with_various_categories():
    """Test natural language generation using priority-based ordering.

    Ordering principle: [Modifier] [Core]
    - Core priority (rightmost/noun): Demographic > Genre > Theme > Cast
    - Modifier priority (leftmost/adjective): Technical > Setting > Cast > Theme > Genre

    Examples based on the specification:
    - "Action Shounen" (Genre + Demographic)
    - "Historical Drama" (Setting + Genre)
    - "Vampire Fantasy" (Cast + Genre)
    - "Isekai Cultivation" (Theme + Theme)
    """
    genres = {
        "Action": {"mood": "hype"},
        "Comedy": {"mood": "funny", "intensity": "light"},
        "Drama": {"mood": "emotional", "intensity": "heavy"},
        "Fantasy": {"mood": "adventurous"},
        "Horror": {"mood": "dark", "intensity": "heavy"},
        "Romance": {"mood": "emotional"},
    }

    namer = ClusterNamer(genres)

    # Genre + Setting: Setting as modifier
    assert namer.name_single_cluster(["Action", "Post-Apocalyptic"]) == "Post-Apocalyptic Action"

    # Genre + Technical: Technical as modifier
    assert namer.name_single_cluster(["Action", "CGI"]) == "CGI Action"

    # Genre + Cast: Cast as modifier
    assert namer.name_single_cluster(["Fantasy", "Vampire"]) == "Vampire Fantasy"

    # Genre + Demographic: Demographic as suffix
    assert namer.name_single_cluster(["Action", "Shounen"]) == "Action Shounen"

    # Genre + Theme: Theme as modifier
    assert namer.name_single_cluster(["Fantasy", "Isekai"]) == "Isekai Fantasy"

    # Setting + Genre with adjectival transform: "Historical Drama"
    assert namer.name_single_cluster(["Drama", "Historical"]) == "Historical Drama"

    # Setting + Genre: "School Romance"
    assert namer.name_single_cluster(["Romance", "School"]) == "School Romance"

    # Theme + Theme: both themes
    assert namer.name_single_cluster(["Isekai", "Cultivation"]) == "Isekai Cultivation"

    # Cast + Setting: Cast has higher priority (4 vs 3), so Setting is modifier
    assert namer.name_single_cluster(["Vampire", "Post-Apocalyptic"]) == "Post-Apocalyptic Vampire"

    # Demographic alone
    assert namer.name_single_cluster(["Shounen"]) == "Shounen"

    # Unknown features (fallback): just joins first two
    assert namer.name_single_cluster(["Unknown1", "Unknown2"]) == "Unknown1 Unknown2"

    # Empty features list
    assert namer.name_single_cluster([]) == ""


def test_cluster_name_adjectival_transformations():
    """Test that plural forms and nouns get proper adjectival transformations."""
    genres = {
        "Action": {"mood": "hype"},
        "Drama": {"mood": "emotional", "intensity": "heavy"},
        "Fantasy": {"mood": "adventurous"},
    }

    namer = ClusterNamer(genres)

    # Plural to singular: "Dragons" → "Dragon"
    assert namer.name_single_cluster(["Dragons", "Fantasy"]) == "Dragon Fantasy"

    # Plural to singular: "Aliens" → "Alien"
    assert namer.name_single_cluster(["Aliens", "Action"]) == "Alien Action"

    # Plural to singular: "Pirates" → "Pirate"
    assert namer.name_single_cluster(["Pirates", "Action"]) == "Pirate Action"

    # Noun to adjective: "Crime" → "Criminal"
    assert namer.name_single_cluster(["Crime", "Drama"]) == "Criminal Drama"

    # Noun to adjective: "Mythology" → "Mythological"
    assert namer.name_single_cluster(["Mythology", "Fantasy"]) == "Mythological Fantasy"

    # Noun to adjective: "Politics" → "Political"
    assert namer.name_single_cluster(["Politics", "Drama"]) == "Political Drama"


def test_cluster_name_edge_cases():
    """Test edge cases: substring overlap, adjective-only pairs, role swapping."""
    genres = {
        "Action": {"mood": "hype"},
        "Comedy": {"mood": "funny", "intensity": "light"},
        "Drama": {"mood": "emotional", "intensity": "heavy"},
        "Fantasy": {"mood": "adventurous"},
    }

    namer = ClusterNamer(genres)

    # Substring overlap: keep longer
    assert namer.name_single_cluster(["Space", "Space Opera"]) == "Space Opera"

    # Substring overlap: keep longer
    assert namer.name_single_cluster(["Urban", "Urban Fantasy"]) == "Urban Fantasy"

    # Both adjective-only with preferred noun
    assert namer.name_single_cluster(["Rural", "Environmental"]) == "Rural Nature"

    # Both adjective-only: "Historical" + "Dystopian" → preferred noun for core
    assert namer.name_single_cluster(["Historical", "Dystopian"]) == "Historical Dystopia"

    # Role swap: core is adjective-only, modifier is noun-capable → swap them
    assert namer.name_single_cluster(["Fantasy", "Historical"]) == "Historical Fantasy"

    # Adjective-only as modifier is fine (no swap needed)
    assert namer.name_single_cluster(["Action", "Dystopian"]) == "Dystopian Action"

    # Both adjective-only, core has no preferred noun but modifier does → swap roles
    namer_swap = ClusterNamer(
        genres={},
    )
    assert namer_swap.name_single_cluster(["Biographical", "Rural"]) == "Biographical Countryside"


def test_cluster_name_adjective_only_fallbacks():
    """Test fallback chain when both features are adjective-only."""
    ClusterNamer.ADJECTIVE_ONLY |= {"FakeAdj1", "FakeAdj2"}

    # Category noun fallback
    namer = ClusterNamer(
        genres={},
    )

    namer.tag_df = pl.DataFrame(
        {
            "name": ["FakeAdj1", "FakeAdj2"],
            "category": ["Theme-Drama", "Setting-Time"],
            "isAdult": [False, False],
        }
    )
    namer.classified_features = namer.classify_features()

    assert namer.name_single_cluster(["FakeAdj1", "FakeAdj2"]) == "FakeAdj2 Drama"

    # Last resort: no preferred noun, no category noun → "Anime"
    # FakeAdj1 is a genre, so category string lookup returns ""
    namer = ClusterNamer(
        genres={"FakeAdj1": {}},
    )

    namer.tag_df = pl.DataFrame(
        {
            "name": ["FakeAdj2"],
            "category": ["Cast-Traits"],
            "isAdult": [False],
        }
    )
    namer.classified_features = namer.classify_features()

    assert namer.name_single_cluster(["FakeAdj1", "FakeAdj2"]) == "FakeAdj2 FakeAdj1 Anime"

    ClusterNamer.ADJECTIVE_ONLY -= {"FakeAdj1", "FakeAdj2"}

    # Adjective-only pair from different categories with 3rd noun-capable candidate
    namer = ClusterNamer(
        genres={},
    )
    # Historical (Setting) + Environmental (Theme) are diverse but both adjective-only
    # → swap Environmental for Vampire (noun-capable)
    name = namer.name_single_cluster(["Historical", "Environmental", "Vampire"])
    assert name == "Historical Vampire"


def test_cast_composition_tags_become_short_modifiers():
    """
    Cast composition tags like 'Primarily Male Cast'
    should be shortened and placed as modifiers.
    """
    namer = ClusterNamer(
        genres={"Horror": {}},
    )

    assert namer.name_single_cluster(["Primarily Male Cast", "Horror"]) == "Male-Led Horror"
    assert namer.name_single_cluster(["Primarily Adult Cast", "Work"]) == "Adult Work"
    assert namer.name_single_cluster(["Primarily Child Cast", "Torture"]) == "Children's Torture"
    assert namer.name_single_cluster(["Yuri", "Primarily Male Cast"]) == "Male-Led Yuri"


def test_adjective_only_core_swaps_with_noun_modifier():
    """When core is adjective-only but modifier is noun-capable, swap them."""
    namer = ClusterNamer(
        genres={"Action": {}},
    )

    # Historical (Setting, adjective-only) has higher core priority than Action (Genre),
    # so it initially becomes core. But since it's adjective-only, it should swap.
    name = namer.name_single_cluster(["Historical", "Action"])

    assert name == "Historical Action"
