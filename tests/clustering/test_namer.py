import polars as pl

from animeippo.clustering.namer import ClusterNamer


def name_single_cluster(namer, features):
    df = pl.DataFrame({"description": [features], "cluster": [0]})

    names = namer.name_clusters(df).select("name")

    if names.height == 1:
        return names.item()

    else:
        return ""


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
    assert name_single_cluster(namer, ["Action", "Post-Apocalyptic"]) == "Post-Apocalyptic Action"

    # Genre + Technical: Technical as modifier
    assert name_single_cluster(namer, ["Action", "CGI"]) == "CGI Action"

    # Genre + Cast: Cast as modifier
    assert name_single_cluster(namer, ["Fantasy", "Vampire"]) == "Vampire Fantasy"

    # Genre + Demographic: Demographic as suffix
    assert name_single_cluster(namer, ["Action", "Shounen"]) == "Action Shounen"

    # Genre + Theme: Theme as modifier
    assert name_single_cluster(namer, ["Fantasy", "Isekai"]) == "Isekai Fantasy"

    # Setting + Genre with adjectival transform: "Historical Drama"
    assert name_single_cluster(namer, ["Drama", "Historical"]) == "Historical Drama"

    # Setting + Genre: "School Romance"
    assert name_single_cluster(namer, ["Romance", "School"]) == "School Romance"

    # Theme + Theme: both themes
    assert name_single_cluster(namer, ["Isekai", "Cultivation"]) == "Isekai Cultivation"

    # Cast + Setting: Cast has higher priority (4 vs 3), so Setting is modifier
    assert name_single_cluster(namer, ["Vampire", "Post-Apocalyptic"]) == "Post-Apocalyptic Vampire"

    # Demographic alone
    assert name_single_cluster(namer, ["Shounen"]) == "Shounen"

    # Unknown features (fallback): just joins first two
    assert name_single_cluster(namer, ["Unknown1", "Unknown2"]) == "Unknown1 Unknown2"

    # Empty features list
    assert name_single_cluster(namer, []) == ""


def test_cluster_name_adjectival_transformations():
    """Test that plural forms and nouns get proper adjectival transformations."""
    genres = {
        "Action": {"mood": "hype"},
        "Drama": {"mood": "emotional", "intensity": "heavy"},
        "Fantasy": {"mood": "adventurous"},
    }

    namer = ClusterNamer(genres)

    # Plural to singular: "Dragons" → "Dragon"
    assert name_single_cluster(namer, ["Dragons", "Fantasy"]) == "Dragon Fantasy"

    # Plural to singular: "Aliens" → "Alien"
    assert name_single_cluster(namer, ["Aliens", "Action"]) == "Alien Action"

    # Plural to singular: "Pirates" → "Pirate"
    assert name_single_cluster(namer, ["Pirates", "Action"]) == "Pirate Action"

    # Noun to adjective: "Crime" → "Criminal"
    assert name_single_cluster(namer, ["Crime", "Drama"]) == "Criminal Drama"

    # Noun to adjective: "Mythology" → "Mythological"
    assert name_single_cluster(namer, ["Mythology", "Fantasy"]) == "Mythological Fantasy"

    # Noun to adjective: "Politics" → "Political"
    assert name_single_cluster(namer, ["Politics", "Drama"]) == "Political Drama"


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
    assert name_single_cluster(namer, ["Space", "Space Opera"]) == "Space Opera"

    # Substring overlap: keep longer
    assert name_single_cluster(namer, ["Urban", "Urban Fantasy"]) == "Urban Fantasy"

    # Both adjective-only with preferred noun
    assert name_single_cluster(namer, ["Rural", "Environmental"]) == "Rural Nature"

    # Both adjective-only: "Historical" + "Dystopian" → preferred noun for core
    assert name_single_cluster(namer, ["Historical", "Dystopian"]) == "Historical Dystopia"

    # Role swap: core is adjective-only, modifier is noun-capable → swap them
    assert name_single_cluster(namer, ["Fantasy", "Historical"]) == "Historical Fantasy"

    # Adjective-only as modifier is fine, no swap needed
    assert name_single_cluster(namer, ["Action", "Dystopian"]) == "Dystopian Action"

    # Both adjective-only, core has no preferred noun but modifier does → swap roles
    assert name_single_cluster(namer, ["Biographical", "Rural"]) == "Biographical Countryside"


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
    namer.classified_features = namer.fit()

    assert name_single_cluster(namer, ["FakeAdj1", "FakeAdj2"]) == "FakeAdj2 Drama"

    ClusterNamer.ADJECTIVE_ONLY -= {"FakeAdj1", "FakeAdj2"}


def test_cast_composition_tags_become_short_modifiers():
    """
    Cast composition tags like 'Primarily Male Cast'
    should be shortened and placed as modifiers.
    """
    namer = ClusterNamer(
        genres={"Horror": {}},
    )

    assert name_single_cluster(namer, ["Primarily Male Cast", "Horror"]) == "Male-Led Horror"
    assert name_single_cluster(namer, ["Primarily Adult Cast", "Work"]) == "Adult Work"
    assert name_single_cluster(namer, ["Primarily Child Cast", "Torture"]) == "Children's Torture"
    assert name_single_cluster(namer, ["Yuri", "Primarily Male Cast"]) == "Male-Led Yuri"


def test_adjective_only_core_swaps_with_noun_modifier():
    """When core is adjective-only but modifier is noun-capable, swap them."""
    namer = ClusterNamer(
        genres={"Action": {}},
    )

    # Historical (Setting, adjective-only) has higher core priority than Action (Genre),
    # so it initially becomes core. But since it's adjective-only, it should swap.
    name = name_single_cluster(namer, ["Historical", "Action"])

    assert name == "Historical Action"
