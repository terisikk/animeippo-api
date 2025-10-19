import os

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
    categorizers = recommender_builder.get_default_categorizers()

    # Verify DebugCategory is first
    assert isinstance(categorizers[0], categories.DebugCategory)


def test_debug_category_is_not_included_when_debug_env_is_false(monkeypatch):
    """Test that DebugCategory is not included when DEBUG=false."""
    monkeypatch.setenv("DEBUG", "false")
    categorizers = recommender_builder.get_default_categorizers()

    # Verify DebugCategory is not in the list
    assert not any(isinstance(cat, categories.DebugCategory) for cat in categorizers)
