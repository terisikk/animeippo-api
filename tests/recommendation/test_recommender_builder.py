from animeippo import providers
from animeippo.recommendation import categories, recommender_builder


class CacheStub:
    def is_available(self):
        return self.is_available


def test_builder_creation_returns_correct_builders():
    assert (
        recommender_builder.build_recommender("anilist").provider.__class__
        == providers.anilist.AniListProvider
    )
    assert (
        recommender_builder.build_recommender("myanimelist").provider.__class__
        == providers.myanimelist.MyAnimeListProvider
    )

    assert (
        recommender_builder.build_recommender("faulty-provider").provider.__class__
        == providers.mixed.MixedProvider
    )


def test_builder_passes_with_or_without_cache(mocker):
    assert (
        recommender_builder.build_recommender("anilist").provider.__class__
        == providers.anilist.AniListProvider
    )

    mocker.patch("animeippo.cache.RedisCache", CacheStub)

    assert (
        recommender_builder.build_recommender("anilist").provider.__class__
        == providers.anilist.AniListProvider
    )


def test_debug_category_is_included_when_debug_env_is_set(monkeypatch):
    """Test that DebugCategory is prepended when DEBUG=true."""
    monkeypatch.setenv("DEBUG", "true")
    categorizers_with_limits = recommender_builder.get_default_categorizers()

    # Verify DebugCategory is first (returns list of (category, top_n) tuples)
    category, top_n = categorizers_with_limits[0]
    assert isinstance(category, categories.DebugCategory)
    assert top_n is None  # Debug category has no limit


def test_debug_category_is_not_included_when_debug_env_is_false(monkeypatch):
    """Test that DebugCategory is not included when DEBUG=false."""
    monkeypatch.setenv("DEBUG", "false")
    categorizers_with_limits = recommender_builder.get_default_categorizers()

    # Verify DebugCategory is not in the list (returns list of (category, top_n) tuples)
    assert not any(isinstance(cat, categories.DebugCategory) for cat, _ in categorizers_with_limits)
