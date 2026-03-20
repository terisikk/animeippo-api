import polars as pl
import pytest

from animeippo.analysis import encoding
from animeippo.profiling import analyser
from animeippo.profiling.model import UserProfile
from animeippo.recommendation.model import RecommendationModel
from tests import test_data, test_provider


def test_profile_analyser_can_run():
    profiler = analyser.ProfileAnalyser(test_provider.AsyncProviderStub())
    profiler.encoder = encoding.CategoricalEncoder()

    categories = profiler.analyse("Janiskeisari")

    assert len(categories) > 0


@pytest.mark.asyncio
async def test_profile_analyser_can_run_when_async_loop_is_already_running():
    profiler = analyser.ProfileAnalyser(test_provider.AsyncProviderStub())
    profiler.encoder = encoding.CategoricalEncoder()

    categories = profiler.analyse("Janiskeisari")

    assert len(categories) > 0


def test_user_profile_can_be_constructred():
    watchlist = pl.DataFrame(test_data.FORMATTED_MAL_USER_LIST)

    user_profile = UserProfile("Test", watchlist)

    assert user_profile.watchlist is not None
    assert user_profile.genre_correlations is not None
    assert user_profile.studio_correlations is not None
    assert user_profile.director_correlations is not None


def test_user_profile_can_be_constructred_with_no_watchlist():
    user_profile = UserProfile("Test", None)

    assert user_profile.watchlist is None
    assert user_profile.genre_correlations is None
    assert user_profile.studio_correlations is None
    assert user_profile.director_correlations is None


def test_user_top_genres_and_tags_can_be_categorized(mocker):
    data = pl.DataFrame(test_data.FORMATTED_ANI_SEASONAL_LIST)
    data = data.with_columns(score=pl.Series([10.0, 8.0]))

    uprofile = UserProfile("Test", data)
    dset = RecommendationModel(uprofile, None)

    profiler = analyser.ProfileAnalyser(None)

    categories = profiler.get_categories(dset)

    assert len(categories) > 0

    uprofile = UserProfile("Test", data)
    uprofile.genre_correlations = pl.DataFrame(
        {"weight": [1, 1], "name": ["Absurd", "Nonexisting"]}
    )
    dset = RecommendationModel(uprofile, None)

    mocker.patch(
        "animeippo.analysis.statistics.weight_categoricals_correlation",
        return_value=pl.DataFrame({"weight": [1, 1], "name": ["Absurd", "Nonexisting"]}),
    )

    categories = profiler.get_categories(dset)
    assert len(categories) > 0

    data = data.drop("genres")
    uprofile = UserProfile("Test", data)

    dset = RecommendationModel(uprofile, None)

    categories = profiler.get_categories(dset)
    assert len(categories) > 0

    data = data.drop("tags")
    uprofile = UserProfile("Test", data)

    dset = RecommendationModel(uprofile, None)

    categories = profiler.get_categories(dset)
    assert len(categories) == 0


def test_clusters_can_be_categorized():
    class ProviderStub:
        def get_nsfw_tags(self):
            return []

        def get_tag_lookup(self):
            return {
                1: {"name": "Shounen", "category": "Demographic", "isAdult": False},
                2: {"name": "Super Power", "category": "Theme-Fantasy", "isAdult": False},
                3: {"name": "Mythology", "category": "Theme-Fantasy", "isAdult": False},
                4: {"name": "Gore", "category": "Theme-Other", "isAdult": False},
            }

        def get_genres(self):
            return {"Action", "Adventure", "Drama", "Fantasy", "Supernatural", "Comedy", "Sci-Fi"}

    data = pl.DataFrame(test_data.FORMATTED_ANI_SEASONAL_LIST)
    data = data.with_columns(
        cluster=pl.Series([0, 1]),
        score=pl.Series([8.0, 9.0]),
        user_status=pl.Series(["completed", "watching"]),
    )

    uprofile = UserProfile("Test", data)
    dset = RecommendationModel(uprofile, None)

    profiler = analyser.ProfileAnalyser(ProviderStub())

    categories = profiler.get_cluster_categories(dset)

    assert len(categories) > 0
    assert "stats" in categories[0]
    assert "count" in categories[0]["stats"]
    assert "mean_score" in categories[0]["stats"]
    assert "completion_rate" in categories[0]["stats"]


def test_cluster_name_generation_with_various_categories():
    """Test natural language generation using priority-based ordering.

    Ordering principle: [Modifier] [Core]
    - Core priority (rightmost/noun): Genre > Theme > Cast-Traits
    - Modifier priority (leftmost/adjective): Demographic > Technical > Setting > Cast > Theme

    Examples based on the specification:
    - "Shounen Action" (Demographic + Genre)
    - "Historical Drama" (Setting + Genre)
    - "Vampire Fantasy" (Cast + Genre)
    - "Isekai Cultivation" (Theme + Theme)
    """
    profiler = analyser.ProfileAnalyser(None)

    # Mock provider
    class MockProvider:
        def get_tag_lookup(self):
            return {
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

        def get_genres(self):
            return {"Horror", "Action", "Comedy", "Fantasy", "Drama", "Romance"}

    profiler.provider = MockProvider()

    # Genre + Setting: Setting as modifier
    name = profiler._generate_natural_cluster_name(["Action", "Post-Apocalyptic"])
    assert name == "Post-Apocalyptic Action"

    # Genre + Technical: Technical as modifier
    name = profiler._generate_natural_cluster_name(["Action", "CGI"])
    assert name == "CGI Action"

    # Genre + Cast: Cast as modifier
    name = profiler._generate_natural_cluster_name(["Fantasy", "Vampire"])
    assert name == "Vampire Fantasy"

    # Genre + Demographic: Demographic as modifier
    name = profiler._generate_natural_cluster_name(["Action", "Shounen"])
    assert name == "Shounen Action"

    # Genre + Theme: Theme as modifier
    name = profiler._generate_natural_cluster_name(["Fantasy", "Isekai"])
    assert name == "Isekai Fantasy"

    # Setting + Genre with adjectival transform: "Historical Drama"
    name = profiler._generate_natural_cluster_name(["Drama", "Historical"])
    assert name == "Historical Drama"

    # Setting + Genre: "School Romance"
    name = profiler._generate_natural_cluster_name(["Romance", "School"])
    assert name == "School Romance"

    # Theme + Theme: both themes
    name = profiler._generate_natural_cluster_name(["Isekai", "Cultivation"])
    assert name == "Isekai Cultivation"

    # Cast + Setting: Cast has higher priority (4 vs 3), so Setting is modifier
    name = profiler._generate_natural_cluster_name(["Vampire", "Post-Apocalyptic"])
    assert name == "Post-Apocalyptic Vampire"

    # Demographic alone
    name = profiler._generate_natural_cluster_name(["Shounen"])
    assert name == "Shounen"

    # Unknown features (fallback): just joins first two
    name = profiler._generate_natural_cluster_name(["Unknown1", "Unknown2"])
    assert name == "Unknown1 Unknown2"

    # Empty features list
    name = profiler._generate_natural_cluster_name([])
    assert name == ""

    # Tag with unknown category: ignored, only recognized feature used
    name = profiler._generate_natural_cluster_name(["Weird Tag", "Action"])
    assert name == "Action"


def test_cluster_name_adjectival_transformations():
    """Test that plural forms and nouns get proper adjectival transformations."""
    profiler = analyser.ProfileAnalyser(None)

    class MockProvider:
        def get_tag_lookup(self):
            return {
                1: {"name": "Dragons", "category": "Cast-Traits", "isAdult": False},
                2: {"name": "Aliens", "category": "Cast-Traits", "isAdult": False},
                3: {"name": "Pirates", "category": "Cast-Traits", "isAdult": False},
                4: {"name": "Crime", "category": "Theme-Other", "isAdult": False},
                5: {"name": "Mythology", "category": "Theme-Other", "isAdult": False},
                6: {"name": "Politics", "category": "Theme-Other", "isAdult": False},
            }

        def get_genres(self):
            return {"Action", "Fantasy", "Drama"}

    profiler.provider = MockProvider()

    # Plural to singular: "Dragons" → "Dragon"
    name = profiler._generate_natural_cluster_name(["Dragons", "Fantasy"])
    assert name == "Dragon Fantasy"

    # Plural to singular: "Aliens" → "Alien"
    name = profiler._generate_natural_cluster_name(["Aliens", "Action"])
    assert name == "Alien Action"

    # Plural to singular: "Pirates" → "Pirate"
    name = profiler._generate_natural_cluster_name(["Pirates", "Action"])
    assert name == "Pirate Action"

    # Noun to adjective: "Crime" → "Criminal"
    name = profiler._generate_natural_cluster_name(["Crime", "Drama"])
    assert name == "Criminal Drama"

    # Noun to adjective: "Mythology" → "Mythological"
    name = profiler._generate_natural_cluster_name(["Mythology", "Fantasy"])
    assert name == "Mythological Fantasy"

    # Noun to adjective: "Politics" → "Political"
    name = profiler._generate_natural_cluster_name(["Politics", "Drama"])
    assert name == "Political Drama"


def test_clusters_can_be_categorized_with_nsfw_filtering():
    """Test that NSFW tags are correctly filtered when categorizing clusters.

    This test ensures that get_nsfw_tags() returns string tag names (not IDs)
    that can be properly filtered from the features column.
    """

    class ProviderStub:
        def get_nsfw_tags(self):
            # Must return tag names (strings), not IDs (integers)
            return {"Bondage", "Hentai", "Explicit Sex"}

        def get_tag_lookup(self):
            return {
                1: {"name": "Action", "category": "Theme-Action", "isAdult": False},
                2: {"name": "Bondage", "category": "Sexual Content", "isAdult": True},
                3: {"name": "Fantasy", "category": "Theme-Fantasy", "isAdult": False},
                4: {"name": "Comedy", "category": "Theme-Comedy", "isAdult": False},
                5: {"name": "Romance", "category": "Theme-Romance", "isAdult": False},
            }

        def get_genres(self):
            return {"Action", "Comedy", "Fantasy", "Romance"}

    # Create data with features that include some NSFW tags
    data = pl.DataFrame(
        {
            "id": [1, 2],
            "title": ["Anime 1", "Anime 2"],
            "features": [["Action", "Bondage", "Fantasy"], ["Comedy", "Romance"]],
            "cluster": [0, 1],
            "score": [8.0, 9.0],
            "user_status": ["completed", "watching"],
        }
    )

    uprofile = UserProfile("Test", data)
    dset = RecommendationModel(uprofile, None)

    profiler = analyser.ProfileAnalyser(ProviderStub())

    # This should not raise a type error about List(Int64) vs Categorical
    categories = profiler.get_cluster_categories(dset)

    assert len(categories) > 0


def test_correlations_are_consistent():
    data = pl.DataFrame(test_data.FORMATTED_ANI_SEASONAL_LIST)
    data = data.with_columns(score=pl.Series([10.0, 8.0]))

    uprofile = UserProfile("Test", data)

    previous = uprofile.get_director_correlations()

    for _ in range(0, 10):
        actual = uprofile.get_director_correlations()
        assert actual.item(0, "weight") == previous.item(0, "weight")
        assert actual.item(0, "name") == previous.item(0, "name")
        assert actual is not None
        previous = actual


def test_favourite_source_calculation():
    data = pl.DataFrame(test_data.FORMATTED_ANI_USER_LIST)
    data = data.with_columns(score=pl.Series([10.0, 8.0]))

    uprofile = UserProfile("Test", data)

    assert uprofile.get_favourite_source() == "original"


def test_favourite_source_calculation_defaults_to_manga():
    data = pl.DataFrame(
        {
            "title": ["Anime 1", "Anime 2", "Anime 3"],
            "score": [9.0, 8.5, 8.0],
            "source": [None, None, None],
        }
    )

    uprofile = UserProfile("Test", data)

    assert uprofile.get_favourite_source() == "manga"
