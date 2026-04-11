from animeippo.profiling.cluster_namer import ClusterNamer


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
    tag_lookup = {
        1: {"name": "Post-Apocalyptic", "category": "Setting-Universe", "isAdult": False},
        2: {"name": "CGI", "category": "Technical", "isAdult": False},
        3: {"name": "Vampire", "category": "Cast-Traits", "isAdult": False},
        4: {"name": "Shounen", "category": "Demographic", "isAdult": False},
        5: {"name": "School", "category": "Setting-Scene", "isAdult": False},
        6: {"name": "Isekai", "category": "Theme-Fantasy", "isAdult": False},
        7: {"name": "Cultivation", "category": "Theme-Fantasy", "isAdult": False},
        8: {"name": "Historical", "category": "Setting-Time", "isAdult": False},
        9: {"name": "Weird Tag", "category": "Unknown-Category", "isAdult": False},
    }
    genres = {"Horror", "Action", "Comedy", "Fantasy", "Drama", "Romance"}
    namer = ClusterNamer(tag_lookup, genres)

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

    # Tag with unknown category: ignored, only recognized feature used
    assert namer.name_single_cluster(["Weird Tag", "Action"]) == "Action"


def test_cluster_name_adjectival_transformations():
    """Test that plural forms and nouns get proper adjectival transformations."""
    tag_lookup = {
        1: {"name": "Dragons", "category": "Cast-Traits", "isAdult": False},
        2: {"name": "Aliens", "category": "Cast-Traits", "isAdult": False},
        3: {"name": "Pirates", "category": "Cast-Traits", "isAdult": False},
        4: {"name": "Crime", "category": "Theme-Other", "isAdult": False},
        5: {"name": "Mythology", "category": "Theme-Other", "isAdult": False},
        6: {"name": "Politics", "category": "Theme-Other", "isAdult": False},
    }
    genres = {"Action", "Fantasy", "Drama"}
    namer = ClusterNamer(tag_lookup, genres)

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
    tag_lookup = {
        1: {"name": "Space", "category": "Setting-Universe", "isAdult": False},
        2: {"name": "Space Opera", "category": "Theme-Sci-Fi", "isAdult": False},
        3: {"name": "Rural", "category": "Setting-Scene", "isAdult": False},
        4: {"name": "Environmental", "category": "Theme-Other", "isAdult": False},
        5: {"name": "Historical", "category": "Setting-Time", "isAdult": False},
        6: {"name": "Urban Fantasy", "category": "Setting-Universe", "isAdult": False},
        7: {"name": "Urban", "category": "Setting-Scene", "isAdult": False},
        8: {"name": "Dystopian", "category": "Setting-Time", "isAdult": False},
    }
    genres = {"Action", "Fantasy", "Drama", "Comedy"}
    namer = ClusterNamer(tag_lookup, genres)

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
        tag_lookup={
            1: {"name": "Biographical", "category": "Theme-Other", "isAdult": False},
            2: {"name": "Rural", "category": "Setting-Scene", "isAdult": False},
        },
        genres=set(),
    )
    assert namer_swap.name_single_cluster(["Biographical", "Rural"]) == "Biographical Countryside"


def test_cluster_name_adjective_only_fallbacks():
    """Test fallback chain when both features are adjective-only."""
    ClusterNamer.ADJECTIVE_ONLY |= {"FakeAdj1", "FakeAdj2"}

    # Category noun fallback
    namer = ClusterNamer(
        tag_lookup={
            1: {"name": "FakeAdj1", "category": "Theme-Drama", "isAdult": False},
            2: {"name": "FakeAdj2", "category": "Setting-Time", "isAdult": False},
        },
        genres=set(),
    )
    assert namer.name_single_cluster(["FakeAdj1", "FakeAdj2"]) == "FakeAdj2 Drama"

    # Last resort: no preferred noun, no category noun → "Anime"
    # FakeAdj1 is a genre (not in tag_lookup), so category string lookup returns ""
    namer = ClusterNamer(
        tag_lookup={1: {"name": "FakeAdj2", "category": "Cast-Traits", "isAdult": False}},
        genres={"FakeAdj1"},
    )
    assert namer.name_single_cluster(["FakeAdj1", "FakeAdj2"]) == "FakeAdj2 FakeAdj1 Anime"

    ClusterNamer.ADJECTIVE_ONLY -= {"FakeAdj1", "FakeAdj2"}

    # Adjective-only pair from different categories with 3rd noun-capable candidate
    namer = ClusterNamer(
        tag_lookup={
            1: {"name": "Historical", "category": "Setting-Time", "isAdult": False},
            2: {"name": "Environmental", "category": "Theme-Other", "isAdult": False},
            3: {"name": "Vampire", "category": "Cast-Traits", "isAdult": False},
        },
        genres=set(),
    )
    # Historical (Setting) + Environmental (Theme) are diverse but both adjective-only
    # → swap Environmental for Vampire (noun-capable)
    name = namer.name_single_cluster(["Historical", "Environmental", "Vampire"])
    assert name == "Historical Vampire"


def test_deprioritize_shared_features():
    """Shared features are moved to the end of the list."""
    namer = ClusterNamer(tag_lookup={}, genres=set())

    result = namer.deprioritize_shared(["Action", "Drama", "Samurai", "Gore"], {"Action", "Drama"})
    assert result == ["Samurai", "Gore", "Action", "Drama"]

    # No shared features → unchanged
    result = namer.deprioritize_shared(["Action", "Drama"], set())
    assert result == ["Action", "Drama"]


def test_duplicate_cluster_names_are_resolved():
    """When two clusters produce the same name, shared features are deprioritized."""
    namer = ClusterNamer(
        tag_lookup={
            1: {"name": "Samurai", "category": "Theme-Action", "isAdult": False},
            2: {"name": "Detective", "category": "Theme-Other", "isAdult": False},
            3: {"name": "Swordplay", "category": "Theme-Action", "isAdult": False},
        },
        genres={"Action", "Drama", "Fantasy"},
    )

    # Clusters 0 and 1 share the same top-2 but differ in later features
    # Both initially produce "Samurai Action", but after dedup they diverge
    cluster_features = {
        "0": ["Action", "Samurai", "Drama"],
        "1": ["Action", "Samurai", "Detective"],
        "2": ["Fantasy", "Swordplay"],
    }

    names = namer.name_clusters(cluster_features)

    # Clusters 0 and 1 should no longer share a name
    assert names["0"] != names["1"]
    # Cluster 2 is unaffected
    assert names["2"] == "Swordplay Fantasy"


def test_deduplicate_no_duplicates_is_noop():
    """When no names collide, all names are returned unchanged."""
    namer = ClusterNamer(
        tag_lookup={
            1: {"name": "Samurai", "category": "Theme-Action", "isAdult": False},
            2: {"name": "Detective", "category": "Theme-Other", "isAdult": False},
        },
        genres={"Action", "Fantasy"},
    )

    cluster_features = {
        "0": ["Action", "Samurai"],
        "1": ["Fantasy", "Detective"],
    }

    names = namer.name_clusters(cluster_features)

    assert names["0"] == "Samurai Action"
    assert names["1"] == "Detective Fantasy"


def test_cast_composition_tags_become_short_modifiers():
    """Cast composition tags like 'Primarily Male Cast' should be shortened and placed as modifiers."""
    namer = ClusterNamer(
        tag_lookup={
            1: {"name": "Primarily Male Cast", "category": "Cast-Main Cast", "isAdult": False},
            2: {"name": "Primarily Adult Cast", "category": "Cast-Main Cast", "isAdult": False},
            3: {"name": "Primarily Child Cast", "category": "Cast-Main Cast", "isAdult": False},
            4: {"name": "Torture", "category": "Theme-Other", "isAdult": False},
            5: {"name": "Work", "category": "Setting-Scene", "isAdult": False},
        },
        genres={"Horror", "Yuri"},
    )

    assert namer.name_single_cluster(["Primarily Male Cast", "Horror"]) == "Male-Led Horror"
    assert namer.name_single_cluster(["Primarily Adult Cast", "Work"]) == "Adult Work"
    assert namer.name_single_cluster(["Primarily Child Cast", "Torture"]) == "Children's Torture"
    assert namer.name_single_cluster(["Yuri", "Primarily Male Cast"]) == "Male-Led Yuri"


def test_adjective_only_core_swaps_with_noun_modifier():
    """When core is adjective-only but modifier is noun-capable, swap them."""
    namer = ClusterNamer(
        tag_lookup={
            1: {"name": "Historical", "category": "Setting-Time", "isAdult": False},
        },
        genres={"Action"},
    )

    # Historical (Setting, adjective-only) has higher core priority than Action (Genre),
    # so it initially becomes core. But since it's adjective-only, it should swap.
    name = namer.name_single_cluster(["Historical", "Action"])

    assert name == "Historical Action"
