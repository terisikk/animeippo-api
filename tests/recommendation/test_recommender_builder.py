from animeippo import providers
from animeippo.recommendation import categories, recommender_builder


class CacheStub:
    def is_available(self):
        return False


def test_builder_creation_returns_correct_builders():
    assert (
        recommender_builder.build_recommender("anilist").provider.__class__
        == providers.anilist.AniListProvider
    )

    assert (
        recommender_builder.build_recommender("mixed").provider.__class__
        == providers.mixed.MixedProvider
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


def test_builder_with_available_cache(mocker):
    class AvailableCacheStub:
        def is_available(self):
            return True

        def get_json(self, key):
            return None

    mocker.patch("animeippo.cache.RedisCache", AvailableCacheStub)

    assert (
        recommender_builder.build_recommender("anilist").provider.__class__
        == providers.anilist.AniListProvider
    )


def test_debug_category_is_included_when_debug_env_is_set(monkeypatch):
    """Test that DebugCategory is prepended in all layouts when DEBUG=true."""
    monkeypatch.setenv("DEBUG", "true")
    layouts = recommender_builder.get_default_categorizers()

    for layout_name, layout in layouts.items():
        category, top_n = layout[0]
        assert isinstance(category, categories.DebugCategory), f"Missing in {layout_name}"
        assert top_n is None


def test_debug_category_is_not_included_when_debug_env_is_false(monkeypatch):
    """Test that DebugCategory is not in any layout when DEBUG=false."""
    monkeypatch.setenv("DEBUG", "false")
    layouts = recommender_builder.get_default_categorizers()

    for layout_name, layout in layouts.items():
        assert not any(isinstance(cat, categories.DebugCategory) for cat, _ in layout), (
            f"Found in {layout_name}"
        )
